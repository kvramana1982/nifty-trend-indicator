"""
Attempt to export the trained models to ONNX so .NET can consume them.
Note: requires skl2onnx and onnxmltools; this step can fail on some Windows setups.
If failure happens, run inference in Python or use joblib models with Python service.
"""
import os
import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASS_MODEL = os.path.join(MODELS_DIR, "lgbm_class.joblib")
REG_MODEL = os.path.join(MODELS_DIR, "lgbm_reg.joblib")

def export_to_onnx(model_path, output_path, n_features):
    model = joblib.load(model_path)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    print("Written ONNX:", output_path)

if __name__ == "__main__":
    # set number of features used in training (must match FEATURES_TO_USE in train.py)
    n_features = 17  # change if you updated FEATURES_TO_USE
    export_to_onnx(CLASS_MODEL, os.path.join(MODELS_DIR, "lgbm_class.onnx"), n_features)
    export_to_onnx(REG_MODEL, os.path.join(MODELS_DIR, "lgbm_reg.onnx"), n_features)
