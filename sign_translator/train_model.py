from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_model(input_csv: str, model_out: str) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Dataset not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "label" not in df.columns:
        raise ValueError("Dataset must include a 'label' column.")

    x = df.drop(columns=["label"]).values
    y = df["label"].values

    if len(set(y)) < 2:
        raise ValueError("Need at least 2 unique gesture labels to train a classifier.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True)),
        ]
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    print("Validation report:")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Saved trained model to {model_out}")
