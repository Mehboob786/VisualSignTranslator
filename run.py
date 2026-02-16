import argparse
import sys


def _check_python_version() -> None:
    # MediaPipe wheels are not available for all Python releases.
    if not ((3, 10) <= sys.version_info[:2] <= (3, 11)):
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise SystemExit(
            "Unsupported Python version: "
            f"{version}. Use Python 3.10 or 3.11 for this project."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisualSignTranslator translator CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    infer_pretrained = sub.add_parser(
        "infer-pretrained",
        help="Run real-time sign inference using pretrained MediaPipe + gesture rules",
    )
    infer_pretrained.add_argument("--camera", type=int, default=0, help="Camera index")
    infer_pretrained.add_argument("--window", type=int, default=12, help="Smoothing window size")
    infer_pretrained.add_argument(
        "--min-stable",
        type=int,
        default=8,
        help="Minimum stable frames before accepting a sign token",
    )
    infer_pretrained.add_argument(
        "--cooldown",
        type=int,
        default=18,
        help="Frames to wait before accepting repeated same token",
    )

    return parser


def main() -> None:
    _check_python_version()

    from sign_translator.pretrained import run_pretrained_inference

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "infer-pretrained":
        run_pretrained_inference(
            camera_index=args.camera,
            window_size=args.window,
            min_stable_frames=args.min_stable,
            repeat_cooldown_frames=args.cooldown,
        )


if __name__ == "__main__":
    main()
