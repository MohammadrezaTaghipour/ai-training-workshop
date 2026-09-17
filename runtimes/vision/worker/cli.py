"""CLI for YOLO image/video detection (OpenCV 5 / Ultralytics env)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_model(model: str):
    from ultralytics import YOLO

    return YOLO(model)


def cmd_detect_image(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    kwargs: dict = {"source": args.source, "conf": args.conf, "save": False}
    if args.classes:
        kwargs["classes"] = [int(c) for c in args.classes.split(",")]

    results = model.predict(**kwargs)
    result = results[0]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(out))

    summary = {
        "source": args.source,
        "output": str(out),
        "model": args.model,
        "boxes": int(len(result.boxes)) if result.boxes is not None else 0,
        "names": result.names,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Saved annotated image → {out} ({summary['boxes']} boxes)")
    return 0


def cmd_detect_video(args: argparse.Namespace) -> int:
    """Process a video file offline and write an annotated MP4."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Cannot open video: {args.source}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    predict_kwargs: dict = {"stream": False, "conf": args.conf, "verbose": False}
    if args.classes:
        predict_kwargs["classes"] = [int(c) for c in args.classes.split(",")]

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, **predict_kwargs)
        annotated = results[0].plot()
        writer.write(annotated)
        frames += 1
        if args.max_frames and frames >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"Saved annotated video → {out_path} ({frames} frames)")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    import torch
    import ultralytics

    print(f"ultralytics {ultralytics.__version__}")
    print(f"torch {torch.__version__}")
    print(f"cuda_available {torch.cuda.is_available()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-lab-vision",
        description="YOLO worker for the AI Learning Lab (isolated vision runtime)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser("detect-image", help="Run detection on an image")
    p_img.add_argument("source", help="Input image path")
    p_img.add_argument("-o", "--output", default="output/detect.jpg", help="Annotated image path")
    p_img.add_argument("-m", "--model", default="yolo12n.pt", help="Model weights path or name")
    p_img.add_argument("--conf", type=float, default=0.25)
    p_img.add_argument("--classes", default="", help="Comma-separated class ids, e.g. 0,2")
    p_img.add_argument("--json", action="store_true", help="Print JSON summary")
    p_img.set_defaults(func=cmd_detect_image)

    p_vid = sub.add_parser("detect-video", help="Run detection on a video file (offline)")
    p_vid.add_argument("source", help="Input video path")
    p_vid.add_argument("-o", "--output", default="output/detect.mp4")
    p_vid.add_argument("-m", "--model", default="yolo12n.pt")
    p_vid.add_argument("--conf", type=float, default=0.25)
    p_vid.add_argument("--classes", default="")
    p_vid.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    p_vid.set_defaults(func=cmd_detect_video)

    p_info = sub.add_parser("info", help="Print runtime versions")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
