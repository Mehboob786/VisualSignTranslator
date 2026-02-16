from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from sign_translator.pretrained import predict_pretrained_from_bgr_frame


def check_python_version() -> None:
    if not ((3, 10) <= sys.version_info[:2] <= (3, 11)):
        st.error(
            f"Unsupported Python version: {sys.version_info.major}.{sys.version_info.minor}. "
            "Use Python 3.10 or 3.11."
        )
        st.stop()


def run_cli(args: List[str]) -> tuple[int, str]:
    cmd = [sys.executable, "run.py", *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.strip()


def tokenize_text(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok]


def text_to_known_signs(text: str, known_labels: List[str]) -> tuple[List[str], List[str]]:
    labels = {lbl.lower() for lbl in known_labels}
    tokens = tokenize_text(text)
    known = [t for t in tokens if t in labels]
    unknown = [t for t in tokens if t not in labels]
    return known, unknown


def main() -> None:
    check_python_version()

    st.set_page_config(page_title="VisualSignTranslator", layout="wide")
    st.title("VisualSignTranslator")
    st.caption("Interactive pre-trained sign translation (no custom model training required).")

    with st.sidebar:
        st.header("Settings")
        camera_index = st.number_input("Camera index", min_value=0, value=0, step=1)
        window_size = st.slider("Smoothing window", 5, 30, 12, 1)
        min_stable = st.slider("Min stable frames", 2, 20, 8, 1)
        cooldown = st.slider("Repeat cooldown frames", 1, 40, 18, 1)
        st.caption("Using MediaPipe pre-trained hand landmarks + built-in gesture rules.")

    known_labels = ["HELLO", "YES", "NO", "POINT", "GOOD"]

    tab_translate = st.tabs(["Generate / Translate (Pre-trained)"])[0]

    with tab_translate:
        st.subheader("Translate Sign to Text")
        st.write("Capture one frame from your webcam and run pre-trained sign prediction.")
        camera_image = st.camera_input("Take a picture of the hand sign")
        if camera_image is not None:
            pil_img = Image.open(io.BytesIO(camera_image.getvalue())).convert("RGB")
            frame_rgb = np.array(pil_img)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            label, conf, annotated = predict_pretrained_from_bgr_frame(frame_bgr)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Prediction: {label} ({conf:.2f})")
            st.markdown(f"**Recognized text:** `{label}`")
            st.caption(f"Confidence: {conf:.2f}")

        st.divider()
        st.subheader("Generate Sign Plan from Text")
        st.write("Type text and this maps words to the available pre-trained gesture labels.")

        phrase = st.text_input("Enter text", value="hello thanks")
        if phrase:
            known, unknown = text_to_known_signs(phrase, known_labels)
            st.write("Known sign tokens:", " ".join(known) if known else "(none)")
            st.write("Unknown tokens:", " ".join(unknown) if unknown else "(none)")
            st.caption(f"Available labels: {', '.join(sorted(known_labels))}")

        st.divider()
        st.subheader("Live Translation")
        st.write("Launch real-time pre-trained translation in OpenCV window (q: quit, c: clear sentence).")
        if st.button("Start Live Translator"):
            code, output = run_cli(
                [
                    "infer-pretrained",
                    "--camera",
                    str(int(camera_index)),
                    "--window",
                    str(int(window_size)),
                    "--min-stable",
                    str(int(min_stable)),
                    "--cooldown",
                    str(int(cooldown)),
                ]
            )
            if code == 0:
                st.success("Live translation ended.")
            else:
                st.error("Live translation failed.")
            if output:
                st.code(output)

if __name__ == "__main__":
    main()
