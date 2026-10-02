import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import fbeta_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "test_holdout.csv"   # using the test holdout split from the original dataset
MODEL_PATH = BASE_DIR / "models" / "decision_tree_model.pkl"


def test_model_quality_on_holdout_set():
    """Protect the model's minimum agreed performance on the fixed holdout split."""
    if not DATA_PATH.exists():
        raise AssertionError(
            f"Evaluation dataset not found at {DATA_PATH}. "
            "Add the project's training dataset before running the ML quality tests."
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns="class")
    y = df["class"].map({"good": 0, "bad": 1})

    assert y.notna().all(), "Unexpected target values found in the dataset."

    # Reproduce the exact holdout split used during training.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    f2 = fbeta_score(y_test, predictions, beta=2)
    recall = recall_score(y_test, predictions)

    # Project acceptance gates. Change these only when the business/model
    # requirements are intentionally changed.
    MIN_F2 = 0.60
    MIN_RECALL = 0.80

    assert f2 >= MIN_F2, (
        f"Model F2 regression: got {f2:.3f}, required >= {MIN_F2:.3f}"
    )
    assert recall >= MIN_RECALL, (
        f"Model recall regression: got {recall:.3f}, required >= {MIN_RECALL:.3f}"
    )
