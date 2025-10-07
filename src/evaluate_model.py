import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def main(data_path, model_path):
    print("📥 Loading data and model...")
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    label_col = [c for c in df.columns if c.lower() == "label"][0]
    X = df.drop(columns=[label_col])
    y = df[label_col]

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"\n✅ Overall Accuracy: {acc:.3f}\n")
    print(classification_report(y, preds))

    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - Full Dataset Evaluation")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.data, args.model)
