import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw


CAMERA_ORDER = [
    "camera_left",
    "camera_top",
    "camera_right",
    "camera_front",
    "camera_left_wrist",
    "camera_right_wrist",
    "camera_front_external",
    "camera_handeye",
    "camera_left_external",
    "camera_right_external",
    "camera_chest",
    "camera_head",
]

BGR_EMBODIMENTS = {
    "h5_franka_3rgb",
    "h5_franka_1rgb",
    "h5_ur_1rgb",
    "h5_franka_fr3_dual",
}

RGB_EMBODIMENTS = {
    "h5_agilex_3rgb",
    "h5_tienkung_gello_1rgb",
    "h5_tienkung_xsens_1rgb",
    "h5_simulation",
    "h5_sim_franka_3rgb",
    "h5_sim_tienkung_1rgb",
    "h5_tienkung_prod1_gello_1rgb",
}


def sorted_cameras(cameras):
    return sorted(
        cameras,
        key=lambda cam: CAMERA_ORDER.index(cam)
        if cam in CAMERA_ORDER
        else len(CAMERA_ORDER),
    )


def infer_embodiment(path, cameras):
    parts = set(path.parts)
    for embodiment in BGR_EMBODIMENTS | RGB_EMBODIMENTS:
        if embodiment in parts:
            return embodiment
    if {"camera_left", "camera_top", "camera_right"}.issubset(cameras):
        return "h5_franka_3rgb"
    if {"camera_front", "camera_left_wrist", "camera_right_wrist"}.issubset(cameras):
        return "h5_agilex_3rgb"
    return "unknown"


def as_uint8_array(value):
    if isinstance(value, (bytes, bytearray)):
        return np.frombuffer(value, dtype=np.uint8)
    return np.asarray(value, dtype=np.uint8)


def decode_image(value, bgr2rgb):
    arr = as_uint8_array(value)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        if arr.size == 2764800:
            image = arr.reshape(720, 1280, 3)
        elif arr.size == 921600:
            image = arr.reshape(480, 640, 3)
        else:
            raise ValueError(f"Cannot decode image buffer with {arr.size} bytes")

    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def make_tile(image, label, tile_size):
    max_w, max_h = tile_size
    pil = Image.fromarray(image)
    pil.thumbnail((max_w, max_h - 30), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", tile_size, "white")
    canvas.paste(pil, ((max_w - pil.width) // 2, 24))
    ImageDraw.Draw(canvas).text((8, 5), label, fill=(0, 0, 0))
    return canvas


def frame_indices(frame_count):
    if frame_count <= 1:
        return [0]
    return sorted({0, frame_count // 2, frame_count - 1})


def inspect_file(path, root, output_dir, tile_size):
    result = {
        "file": str(path.relative_to(root)),
        "status": "ok",
        "cameras": [],
        "frames": None,
        "embodiment": None,
        "samples": [],
    }

    try:
        with h5py.File(path, "r") as h5:
            if "observations" not in h5 or "rgb_images" not in h5["observations"]:
                result["status"] = "missing observations/rgb_images"
                return result

            rgb_group = h5["observations"]["rgb_images"]
            cameras = sorted_cameras(list(rgb_group.keys()))
            result["cameras"] = cameras
            result["frames"] = min(int(rgb_group[camera].shape[0]) for camera in cameras)
            result["embodiment"] = infer_embodiment(path, set(cameras))
            bgr2rgb = result["embodiment"] in BGR_EMBODIMENTS

            episode = path.parts[-3]
            for frame in frame_indices(result["frames"]):
                tiles = []
                for camera in cameras:
                    image = decode_image(rgb_group[camera][frame], bgr2rgb=bgr2rgb)
                    tiles.append(make_tile(image, f"{camera} frame={frame}", tile_size))

                sheet = Image.new(
                    "RGB", (tile_size[0] * len(tiles), tile_size[1]), "white"
                )
                for index, tile in enumerate(tiles):
                    sheet.paste(tile, (index * tile_size[0], 0))

                output = output_dir / f"{episode}_frame_{frame:04d}_cameras.jpg"
                sheet.save(output, quality=92)
                result["samples"].append(str(output.relative_to(root)))
    except Exception as exc:
        result["status"] = f"error: {exc}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Inspect RoboMIND HDF5 camera streams and export camera contact sheets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing RoboMIND files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write contact sheets. Defaults to <root>/samples.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Where to write JSON manifest. Defaults to <root>/crossview_manifest.json.",
    )
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=270)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "samples").resolve()
    manifest = (args.manifest or root / "crossview_manifest.json").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path in sorted(root.rglob("*.hdf5")):
        if "__MACOSX" in path.parts:
            continue
        results.append(
            inspect_file(
                path=path,
                root=root,
                output_dir=output_dir,
                tile_size=(args.tile_width, args.tile_height),
            )
        )

    manifest.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for item in results:
        print(
            f"{item['file']}\tstatus={item['status']}\t"
            f"frames={item['frames']}\tcameras={item['cameras']}"
        )
        for sample in item["samples"]:
            print(f"  sample={sample}")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
