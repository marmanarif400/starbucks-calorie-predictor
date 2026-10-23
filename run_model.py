
import pickle
from pathlib import Path

import sklearn

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "starbucks_decision_tree.pkl"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with MODEL_PATH.open("rb") as file:
        artifact = pickle.load(file)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    print("Starbucks Decision Tree loaded successfully!")
    print("Model type:", artifact.get("model_type", type(model).__name__))
    print("Scikit-learn version:", sklearn.__version__)
    print("\nFeatures required for prediction:")

    for feature in feature_columns:
        print(f"- {feature}")

    print("\nThe model is ready to receive compatible input data.")

if __name__ == "__main__":
    main()
