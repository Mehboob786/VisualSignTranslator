# VisualSignTranslator

Interactive sign language translator prototype built with Python, OpenCV, MediaPipe, and Streamlit.

This application focuses only on translation functionality:
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

## Run App

```bash
streamlit run app.py
```

## Translator Features

- Snapshot translation:
  - Capture a hand sign image and get predicted text below the image.
- Live translation:
  - Launch real-time webcam translation window.
  - Controls: `q` to quit, `c` to clear sentence buffer.
- Text-to-sign mapping:
  - Maps words to supported labels: `HELLO`, `YES`, `NO`, `POINT`, `GOOD`.

## Optional CLI Live Translator

```bash
python run.py infer-pretrained --camera 0 --window 12 --min-stable 8 --cooldown 18
```
