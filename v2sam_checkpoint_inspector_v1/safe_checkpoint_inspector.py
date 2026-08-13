#!/usr/bin/env python3
"""Statically inspect a protocol-2 PyTorch ZIP checkpoint.

This program never calls pickle.load, torch.load, GLOBAL callables, REDUCE
callables, or object serialization hooks. It interprets the small opcode
subset used by the audited V2-SAM checkpoint and turns tensors into immutable
descriptors.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import pickletools
import sys
import zipfile
from pathlib import Path
from typing import Any


AUDITED_GLOBALS = {
    "__builtin__.getattr",
    "_codecs.encode",
    "collections.OrderedDict",
    "mmengine.logging.history_buffer.HistoryBuffer",
    "numpy.dtype",
    "numpy.ndarray",
    "numpy._core.multiarray._reconstruct",
    "numpy._core.multiarray.scalar",
    "torch.FloatStorage",
    "torch._utils._rebuild_tensor_v2",
}


class InspectionError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class GlobalRef:
    path: str


@dataclasses.dataclass(frozen=True)
class StorageDesc:
    storage_type: str
    key: str
    location: str
    numel: int


@dataclasses.dataclass(frozen=True)
class TensorDesc:
    storage: StorageDesc
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    requires_grad: bool


@dataclasses.dataclass
class Placeholder:
    kind: str
    value: Any = None
    state: Any = None


@dataclasses.dataclass(frozen=True)
class HistoryMethodRef:
    name: str


def _global_path(argument: Any) -> str:
    if not isinstance(argument, str) or " " not in argument:
        raise InspectionError(f"malformed GLOBAL argument: {argument!r}")
    module, name = argument.split(" ", 1)
    return f"{module}.{name}"


def _as_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise InspectionError(f"{label} is not a tuple")
    if not all(type(item) is int and item >= 0 for item in value):
        raise InspectionError(f"{label} contains invalid dimensions")
    return value


class StaticPickleMachine:
    def __init__(self) -> None:
        self.stack: list[Any] = []
        self.metastack: list[list[Any]] = []
        self.memo: dict[int, Any] = {}
        self.protocol: int | None = None
        self.globals_seen: set[str] = set()

    def pop_mark(self) -> list[Any]:
        if not self.metastack:
            raise InspectionError("MARK stack underflow")
        items = self.stack
        self.stack = self.metastack.pop()
        return items

    def persistent_storage(self, pid: Any) -> StorageDesc:
        if not isinstance(pid, tuple) or len(pid) != 5:
            raise InspectionError(f"invalid persistent id: {pid!r}")
        tag, storage_type, key, location, numel = pid
        if isinstance(tag, bytes):
            tag = tag.decode("ascii", "strict")
        if tag != "storage":
            raise InspectionError(f"unsupported persistent id tag: {tag!r}")
        if not isinstance(storage_type, GlobalRef):
            raise InspectionError("storage type is not a GLOBAL reference")
        if not (
            storage_type.path.startswith("torch.")
            and storage_type.path.endswith("Storage")
        ):
            raise InspectionError(
                f"unsupported storage type: {storage_type.path}"
            )
        if not isinstance(key, str) or not key:
            raise InspectionError("invalid storage key")
        if isinstance(location, bytes):
            location = location.decode("ascii", "strict")
        if not isinstance(location, str):
            raise InspectionError("invalid storage location")
        if type(numel) is not int or numel < 0:
            raise InspectionError("invalid storage size")
        return StorageDesc(storage_type.path, key, location, numel)

    def reduce(self, function: Any, args: Any) -> Any:
        if not isinstance(args, tuple):
            raise InspectionError("REDUCE arguments are not a tuple")
        if not isinstance(function, GlobalRef):
            raise InspectionError(f"REDUCE callable is not symbolic: {function!r}")
        path = function.path

        if path == "collections.OrderedDict":
            if args:
                raise InspectionError("OrderedDict constructor has arguments")
            return collections.OrderedDict()

        if path == "torch._utils._rebuild_tensor_v2":
            if len(args) != 6:
                raise InspectionError("unexpected _rebuild_tensor_v2 arity")
            storage, offset, shape, stride, requires_grad, _hooks = args
            if not isinstance(storage, StorageDesc):
                raise InspectionError("tensor storage is not a storage descriptor")
            if type(offset) is not int or offset < 0:
                raise InspectionError("invalid tensor storage offset")
            shape = _as_int_tuple(shape, "shape")
            stride = _as_int_tuple(stride, "stride")
            if len(shape) != len(stride):
                raise InspectionError("shape/stride rank mismatch")
            if type(requires_grad) is not bool:
                raise InspectionError("requires_grad is not bool")
            return TensorDesc(storage, offset, shape, stride, requires_grad)

        if path == "numpy._core.multiarray._reconstruct":
            return Placeholder("numpy.ndarray")
        if path == "numpy.dtype":
            return Placeholder("numpy.dtype", value=args)
        if path == "numpy._core.multiarray.scalar":
            return Placeholder("numpy.scalar", value=args)
        if path == "_codecs.encode":
            return Placeholder("encoded-bytes", value=args)
        if path == "__builtin__.getattr":
            if len(args) != 2 or not isinstance(args[1], str):
                raise InspectionError("malformed getattr reduction")
            target, name = args
            if not (
                isinstance(target, GlobalRef)
                and target.path
                == "mmengine.logging.history_buffer.HistoryBuffer"
                and name in {"min", "max", "current", "mean"}
            ):
                raise InspectionError(
                    f"getattr is not allowlisted: {target!r}.{name}"
                )
            return HistoryMethodRef(name)

        raise InspectionError(f"unsupported REDUCE callable: {path}")

    def run(self, payload: bytes) -> Any:
        result: Any = None
        for opcode, argument, position in pickletools.genops(payload):
            name = opcode.name

            if name == "PROTO":
                self.protocol = argument
            elif name == "GLOBAL":
                path = _global_path(argument)
                self.globals_seen.add(path)
                if path not in AUDITED_GLOBALS:
                    raise InspectionError(f"unsupported GLOBAL: {path}")
                self.stack.append(GlobalRef(path))
            elif name in {"BINPUT", "LONG_BINPUT"}:
                if not self.stack:
                    raise InspectionError("memo write on empty stack")
                self.memo[argument] = self.stack[-1]
            elif name in {"BINGET", "LONG_BINGET"}:
                if argument not in self.memo:
                    raise InspectionError(f"unknown memo index: {argument}")
                self.stack.append(self.memo[argument])
            elif name == "MARK":
                self.metastack.append(self.stack)
                self.stack = []
            elif name == "EMPTY_DICT":
                self.stack.append({})
            elif name == "EMPTY_LIST":
                self.stack.append([])
            elif name == "EMPTY_TUPLE":
                self.stack.append(())
            elif name == "EMPTY_SET":
                self.stack.append(set())
            elif name == "NONE":
                self.stack.append(None)
            elif name == "NEWTRUE":
                self.stack.append(True)
            elif name == "NEWFALSE":
                self.stack.append(False)
            elif name in {"BININT", "BININT1", "BININT2", "BINFLOAT"}:
                self.stack.append(argument)
            elif name in {"BINUNICODE", "SHORT_BINSTRING"}:
                self.stack.append(argument)
            elif name == "TUPLE":
                items = self.pop_mark()
                self.stack.append(tuple(items))
            elif name == "TUPLE1":
                self.stack[-1:] = [(self.stack[-1],)]
            elif name == "TUPLE2":
                self.stack[-2:] = [(self.stack[-2], self.stack[-1])]
            elif name == "TUPLE3":
                self.stack[-3:] = [
                    (self.stack[-3], self.stack[-2], self.stack[-1])
                ]
            elif name == "APPEND":
                item = self.stack.pop()
                container = self.stack[-1]
                if type(container) is not list:
                    raise InspectionError("APPEND target is not list")
                container.append(item)
            elif name == "APPENDS":
                items = self.pop_mark()
                container = self.stack[-1]
                if type(container) is not list:
                    raise InspectionError("APPENDS target is not list")
                container.extend(items)
            elif name == "SETITEM":
                value = self.stack.pop()
                key = self.stack.pop()
                container = self.stack[-1]
                if not isinstance(container, dict):
                    raise InspectionError("SETITEM target is not dict")
                container[key] = value
            elif name == "SETITEMS":
                items = self.pop_mark()
                container = self.stack[-1]
                if not isinstance(container, dict) or len(items) % 2:
                    raise InspectionError("malformed SETITEMS")
                for index in range(0, len(items), 2):
                    container[items[index]] = items[index + 1]
            elif name == "BINPERSID":
                self.stack.append(self.persistent_storage(self.stack.pop()))
            elif name == "REDUCE":
                args = self.stack.pop()
                function = self.stack[-1]
                self.stack[-1] = self.reduce(function, args)
            elif name == "NEWOBJ":
                args = self.stack.pop()
                cls = self.stack.pop()
                if not isinstance(cls, GlobalRef) or not isinstance(args, tuple):
                    raise InspectionError("malformed NEWOBJ")
                self.stack.append(Placeholder(cls.path, value=args))
            elif name == "BUILD":
                state = self.stack.pop()
                instance = self.stack[-1]
                if isinstance(instance, Placeholder):
                    instance.state = state
                elif isinstance(instance, collections.OrderedDict):
                    if isinstance(state, dict):
                        instance.__dict__.update(state)
                    else:
                        raise InspectionError("invalid OrderedDict state")
                else:
                    raise InspectionError(
                        f"unsupported BUILD target: {type(instance).__name__}"
                    )
            elif name == "STOP":
                if self.metastack or len(self.stack) != 1:
                    raise InspectionError("unbalanced pickle stack at STOP")
                result = self.stack.pop()
                break
            else:
                raise InspectionError(
                    f"unsupported opcode {name} at byte {position}"
                )
        else:
            raise InspectionError("pickle stream has no STOP")
        return result


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InspectionError(f"checkpoint is not a file: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        pickle_names = [
            name for name in names
            if name == "data.pkl" or name.endswith("/data.pkl")
        ]
        if len(pickle_names) != 1:
            raise InspectionError("expected exactly one data.pkl")
        payload = archive.read(pickle_names[0])

    machine = StaticPickleMachine()
    root = machine.run(payload)
    if not isinstance(root, dict):
        raise InspectionError("checkpoint root is not a dictionary")

    state_key = next(
        (
            key for key in ("state_dict", "model", "model_state_dict")
            if isinstance(root.get(key), dict) and root[key]
        ),
        None,
    )
    if state_key is None:
        raise InspectionError("checkpoint has no state dictionary")
    state_dict = root[state_key]
    tensors = {
        str(key): value
        for key, value in state_dict.items()
        if isinstance(value, TensorDesc)
    }
    non_tensor_count = len(state_dict) - len(tensors)
    prefix_counts = collections.Counter(
        key.split(".", 1)[0] for key in tensors
    )
    tensor_report = {
        key: {
            "shape": list(value.shape),
            "stride": list(value.stride),
            "dtype": value.storage.storage_type,
            "storage_key": value.storage.key,
            "storage_numel": value.storage.numel,
            "storage_offset": value.storage_offset,
            "location": value.storage.location,
            "requires_grad": value.requires_grad,
        }
        for key, value in tensors.items()
    }
    structure_bytes = json.dumps(
        tensor_report,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    unknown_globals = sorted(machine.globals_seen - AUDITED_GLOBALS)
    return {
        "static_only": True,
        "checkpoint": str(path.resolve()),
        "checkpoint_bytes": path.stat().st_size,
        "zip_entries": len(names),
        "data_pickle_entry": pickle_names[0],
        "data_pickle_bytes": len(payload),
        "data_pickle_sha256": hashlib.sha256(payload).hexdigest(),
        "pickle_protocol": machine.protocol,
        "globals": sorted(machine.globals_seen),
        "unknown_globals": unknown_globals,
        "state_dict_key": state_key,
        "tensor_count": len(tensors),
        "non_tensor_state_entries": non_tensor_count,
        "prefix_counts": dict(sorted(prefix_counts.items())),
        "tensor_structure_sha256": hashlib.sha256(structure_bytes).hexdigest(),
        "tensors": tensor_report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        report = inspect_checkpoint(args.checkpoint)
    except (InspectionError, OSError, zipfile.BadZipFile) as exc:
        print(f"INSPECTION_FAILED: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(report, sys.stdout, sort_keys=True, separators=(",", ":"))
        sys.stdout.write("\n")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
