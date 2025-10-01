import os
import json
import joblib
import platform
import sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS

# Paths de artefactos
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
MANIFEST_PATH = ARTIFACTS_DIR / "model_card.json"

# Validación de artefactos
if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
    raise FileNotFoundError("Faltan artefactos. Ejecuta antes el notebook 01 (serialización).")

# Cargar modelo y manifiesto
loaded_model = joblib.load(MODEL_PATH)
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

DECLARED_FEATURES = manifest.get("features", []) or ["radius_worst", "perimeter_worst", "area_worst","concavity_worst","concave points_worst"]
API_KEY_REQUIRED = os.getenv("API_KEY")

# ------------------------
# Funciones auxiliares
# ------------------------

def check_api_key(req):
    """Valida la API Key si está configurada en el entorno."""
    if not API_KEY_REQUIRED:
        return None
    key = req.headers.get("X-API-KEY")
    if key != API_KEY_REQUIRED:
        return "API key inválida o ausente."
    return None


def validate_input_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """Verifica que el DataFrame contenga todas las columnas requeridas."""
    missing = [c for c in DECLARED_FEATURES if c not in df_in.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Se esperaban: {DECLARED_FEATURES}")
    return df_in[DECLARED_FEATURES].copy()


def payload_to_dataframe(payload: Any) -> pd.DataFrame:
    """Convierte el payload JSON en un DataFrame válido."""
    if not isinstance(payload, list) or not all(isinstance(r, dict) for r in payload):
        raise TypeError("El payload debe ser una lista de objetos (dict).")
    df = pd.DataFrame(payload)
    _ = validate_input_df(df)
    return df


def predict_batch(df_in: pd.DataFrame, return_proba: bool = True):
    """Genera predicciones y probabilidades (si están disponibles)."""
    X = validate_input_df(df_in)
    yhat = loaded_model.predict(X)
    proba = None
    if return_proba and hasattr(loaded_model, "predict_proba"):
        proba = loaded_model.predict_proba(X)
    return yhat, proba


# ------------------------
# API con Flask
# ------------------------

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    err = check_api_key(request)
    if err:
        return jsonify({"ok": False, "error": err}), 401
    return jsonify({
        "ok": True,
        "python": platform.python_version(),
        "sklearn_loaded": sklearn.__version__,
        "model_path": str(MODEL_PATH.resolve())
    })


@app.get("/model-info")
def model_info():
    err = check_api_key(request)
    if err:
        return jsonify({"ok": False, "error": err}), 401
    
    return jsonify({
        "ok": True,
        "model_name": manifest.get("name"),
        "created_at": manifest.get("created_at"),
        "framework": manifest.get("framework"),
        "sklearn_version_train": manifest.get("sklearn_version"),
        "python_version_train": manifest.get("python_version"),
        "features": manifest.get("features"),
        "target": manifest.get("target"),
        "model_type": manifest.get("model_type"),
        "model_params": manifest.get("model_params"),
        "dataset": manifest.get("dataset"),
        "metrics": manifest.get("metrics"),
        "artifact_path": manifest.get("artifact_path"),
        "dependencies": manifest.get("dependencies")
    })


@app.post("/predict")
def predict():
    err = check_api_key(request)
    if err:
        return jsonify({"ok": False, "error": err}), 401
    try:
        payload = request.get_json(force=True, silent=False)
        df = payload_to_dataframe(payload)
        yhat, proba = predict_batch(df, return_proba=True)

        out = df.copy()
        out["prediction"] = yhat

        if proba is not None and proba.shape[1] >= 2:
            out["p1"] = proba[:, -1].astype(float)

        return jsonify({"ok": True, "result": out.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ------------------------
# Main
# ------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)


