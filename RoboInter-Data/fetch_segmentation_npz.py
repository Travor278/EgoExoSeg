"""Selectively extract single segmentation .npz files from RoboInter-Data's
~53 GB split zip (Annotation_raw/segmentation_npz.zip.00/.01/.02) via HTTP
Range requests -- no need to download the whole archive.

How it works: the three parts are a plain byte-split of ONE zip file
(README: `cat segmentation_npz.zip.* > segmentation_npz.zip`), so absolute
offsets in the concatenated stream are valid zip offsets. We fetch the zip
central directory once (~20 MB, cached to segzip_index.json), then pull just
the compressed bytes of each requested member.

Usage:
    python fetch_segmentation_npz.py 10020_exterior_image_1_left 10020_exterior_image_2_left
Names are matched by basename (with or without .npz). Output goes to
segmentation_npz/<name>.npz next to this script.
"""
from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
BASE = (
    "https://huggingface.co/datasets/InternRobotics/RoboInter-Data/"
    "resolve/main/Annotation_raw/"
)
PARTS = [
    "segmentation_npz.zip.00",
    "segmentation_npz.zip.01",
    "segmentation_npz.zip.02",
]
INDEX_PATH = ROOT / "segzip_index.json"
OUT_DIR = ROOT / "segmentation_npz"

SESSION = requests.Session()


def part_size(name: str) -> int:
    r = SESSION.get(
        BASE + name, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=120
    )
    r.raise_for_status()
    content_range = r.headers.get("content-range", "")
    if "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    raise RuntimeError(f"no content-range for {name}: {dict(r.headers)}")


def read_range(sizes: list[int], abs_start: int, length: int) -> bytes:
    """Read [abs_start, abs_start+length) from the concatenated part stream."""
    starts = []
    acc = 0
    for s in sizes:
        starts.append(acc)
        acc += s
    assert 0 <= abs_start and abs_start + length <= acc, "range outside archive"
    out = bytearray()
    pos, remaining = abs_start, length
    for i, s in enumerate(sizes):
        if remaining <= 0:
            break
        pstart, pend = starts[i], starts[i] + s
        if pos >= pend:
            continue
        local = pos - pstart
        take = min(remaining, s - local)
        r = SESSION.get(
            BASE + PARTS[i],
            headers={"Range": f"bytes={local}-{local + take - 1}"},
            allow_redirects=True,
            timeout=1800,
        )
        if r.status_code not in (200, 206):
            raise RuntimeError(f"range GET failed on {PARTS[i]}: {r.status_code}")
        chunk = r.content
        if len(chunk) != take:
            raise RuntimeError(f"short read {len(chunk)} != {take} on {PARTS[i]}")
        out += chunk
        pos += take
        remaining -= take
    return bytes(out)


def parse_central_directory(cd: bytes, total_entries: int) -> dict[str, list]:
    """Return {member_name: [abs_local_header_offset, compressed_size, method, uncompressed_size]}."""
    entries: dict[str, list] = {}
    off = 0
    for _ in range(total_entries):
        if cd[off : off + 4] != b"PK\x01\x02":
            raise RuntimeError(f"bad central header sig at {off}")
        (
            _vmade,
            _vneed,
            _flags,
            method,
            _mtime,
            _mdate,
            _crc,
            csize,
            usize,
            fnlen,
            exlen,
            cmlen,
            _disk,
            _iattr,
            _eattr,
            lho,
        ) = struct.unpack("<HHHHHHIIIHHHHHII", cd[off + 4 : off + 46])
        name = cd[off + 46 : off + 46 + fnlen].decode("utf-8", "replace")
        extra = cd[off + 46 + fnlen : off + 46 + fnlen + exlen]
        # ZIP64 extra field overrides 0xFFFFFFFF placeholders, in fixed order
        if 0xFFFFFFFF in (csize, usize, lho):
            e = 0
            while e + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[e : e + 4])
                body = extra[e + 4 : e + 4 + hsz]
                if hid == 0x0001:
                    b = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", body[b : b + 8])[0]
                        b += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", body[b : b + 8])[0]
                        b += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", body[b : b + 8])[0]
                        b += 8
                    break
                e += 4 + hsz
        entries[name] = [lho, csize, method, usize]
        off += 46 + fnlen + exlen + cmlen
    return entries


def build_index() -> dict:
    print("probing part sizes ...")
    sizes = [part_size(p) for p in PARTS]
    total = sum(sizes)
    print(f"  parts: {sizes}  total={total:,}")

    tail_len = 65_557 + 20 + 56  # max EOCD+comment + EOCD64 locator + EOCD64
    tail = read_range(sizes, total - tail_len, tail_len)

    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("EOCD not found")
    n_entries, cd_size, cd_off = struct.unpack("<HII", tail[eocd + 10 : eocd + 20])
    loc = tail.rfind(b"PK\x06\x07", 0, eocd)
    if loc >= 0:  # ZIP64
        eocd64_off = struct.unpack("<Q", tail[loc + 8 : loc + 16])[0]
        eocd64 = read_range(sizes, eocd64_off, 56)
        if eocd64[:4] != b"PK\x06\x06":
            raise RuntimeError("bad EOCD64 signature")
        n_entries = struct.unpack("<Q", eocd64[32:40])[0]
        cd_size = struct.unpack("<Q", eocd64[40:48])[0]
        cd_off = struct.unpack("<Q", eocd64[48:56])[0]
    print(f"  central directory: {n_entries:,} entries, {cd_size / 1e6:.1f} MB at offset {cd_off:,}")

    print("downloading central directory ...")
    cd = read_range(sizes, cd_off, cd_size)
    entries = parse_central_directory(cd, n_entries)
    index = {"part_sizes": sizes, "entries": entries}
    INDEX_PATH.write_text(json.dumps(index), encoding="utf-8")
    print(f"  cached index -> {INDEX_PATH} ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")
    return index


def load_index() -> dict:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return build_index()


def extract(index: dict, member: str, out_path: Path) -> Path:
    sizes = index["part_sizes"]
    lho, csize, method, usize = index["entries"][member]
    header = read_range(sizes, lho, 30)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {member}")
    fnlen, exlen = struct.unpack("<HH", header[26:30])
    data = read_range(sizes, lho + 30 + fnlen + exlen, csize)
    if method == 8:
        d = zlib.decompressobj(-15)
        raw = d.decompress(data) + d.flush()
    elif method == 0:
        raw = data
    else:
        raise RuntimeError(f"unsupported compression method {method}")
    if usize and len(raw) != usize:
        raise RuntimeError(f"size mismatch for {member}: {len(raw)} != {usize}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    return out_path


def main(wanted: list[str]) -> None:
    index = load_index()
    entries = index["entries"]
    by_base = {Path(name).name: name for name in entries}
    print(f"index ready: {len(entries):,} members")

    for w in wanted:
        base = w if w.endswith(".npz") else w + ".npz"
        member = by_base.get(base)
        if member is None:
            print(f"!! not in archive: {base}")
            continue
        out = OUT_DIR / base
        if out.exists():
            print(f"already have {out.name}")
            continue
        extract(index, member, out)
        print(f"extracted {member} -> {out} ({out.stat().st_size:,} B)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
