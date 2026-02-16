# VisualSignTranslator

Interactive sign language translator built with Python, OpenCV, MediaPipe, and Streamlit.

This application is translator-only (no model training workflow in the app):
- Sign-to-text from webcam snapshot
- Live sign-to-text translation
- Text-to-sign token mapping for supported labels

## Setup

Python compatibility:
- Use Python `3.10` or `3.11` (MediaPipe is not available on Python `3.14`).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.11` is not installed on macOS:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run App

```bash
streamlit run app.py
```

## Features

- Snapshot translation:
  - Capture a hand sign image.
  - Predicted text is shown below the image.
- Live translation:
  - Launches real-time webcam translation in an OpenCV window.
  - Controls in live window: `q` to quit, `c` to clear sentence buffer.
- Text-to-sign mapping:
  - Maps words to supported labels: `HELLO`, `YES`, `NO`, `POINT`, `GOOD`.

## CLI (Live Translator)

```bash
python run.py infer-pretrained --camera 0 --window 12 --min-stable 8 --cooldown 18
```

Arguments:
- `--camera`: camera index (default `0`)
- `--window`: smoothing window size
- `--min-stable`: frames required before accepting a token
- `--cooldown`: delay before accepting repeated token again

## Notes

- This is a prototype translator for a limited built-in gesture set.
- For best results, use good lighting and keep one hand clearly visible.
