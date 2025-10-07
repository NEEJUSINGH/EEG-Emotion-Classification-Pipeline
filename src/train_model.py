import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
import os

def main(data_path, model_path):
    # ===========================
    # 1️⃣ Load and inspect data
    # ===========================
    print("📥 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded successfully with {df.shape[0]} samples and {df.shape[1]} features.")

    # Automatically detect the label column
    label_col = None
    for col in df.columns:
        if col.lower() in ['label', 'emotion', 'target', 'class']:
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[-1]  # fallback: last column

    print(f"🎯 Using '{label_col}' as target column")

    # Split into features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Verify features are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # ==================================
    # 2️⃣ Train/Test Split
    # ==================================
    print("✂️ Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # ==================================
    # 3️⃣ Train the Model
    # ==================================
    print("🤖 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Model training complete!")

    # ==================================
    # 4️⃣ Evaluate Model
    # ==================================
    print("📊 Evaluating model performance...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n🎯 Accuracy: {acc:.3f}\n")
    print("Detailed classification report:\n")
    print(classification_report(y_test, preds))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix - EEG Emotion Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # ==================================
    # 5️⃣ Feature Importance
    # ==================================
    print("🧠 Visualizing top feature importances...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15 important features
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Importance Score")
    plt.title("Top EEG Feature Importances")
    plt.tight_layout()
    plt.show()

    # ==================================
    # 6️⃣ Save Model
    # ==================================
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model saved successfully to: {model_path}")

    print("\n🎉 Training pipeline completed successfully!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG Emotion Classification Model")
    parser.add_argument("--data", required=True, help="Path to the input CSV data")
    parser.add_argument("--model", required=True, help="Path to save trained model (pkl file)")
    args = parser.parse_args()

    main(args.data, args.model)
