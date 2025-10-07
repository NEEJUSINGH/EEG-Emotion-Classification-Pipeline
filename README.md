# 🧠 EEG Emotion Classification Pipeline  
*A Machine Learning approach to decoding emotional states from brain signals*

---

## Overview

This project applies **Artificial Intelligence** and **Signal Processing** techniques to **Electroencephalogram (EEG)** data to classify human emotional states such as *Calm*, *Neutral*, or *Stressed*.  

By combining **Fourier Transform–based feature extraction** and **machine learning models**, the project demonstrates how raw brainwave data can be transformed into interpretable patterns of human emotion.  

The pipeline is modular, reproducible, and designed to reflect the kind of data fusion and analysis used in **Translational Neural Engineering** and **Cognitive Neuroscience** research.

---

## Objectives

- Build a **reproducible data analysis pipeline** for brain signal data  
- Apply **frequency-domain (Fourier/Wavelet)** transformations to EEG signals  
- Train and evaluate **machine learning models** for emotion prediction  
- Visualize brainwave activity across different emotional states  
- Demonstrate an end-to-end workflow suitable for **neuroscience AI applications**

---

## Project Structure



```bash

EEG-Emotion-Classifier/
│
├── data/               # sample EEG .csv files
├── notebooks/          # exploration and plots
├── src/
│   ├── preprocess.py   # filtering, FFT, feature extraction
│   ├── train_model.py  # train/test split, classifier
│   ├── evaluate.py     # confusion matrix, metrics
│   └── visualize.py    # band power plots
├── requirements.txt
└── README.md

```

---

## Dataset

- **Source:** [Kaggle – EEG Brainwave Dataset: Feeling Emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions)  
- **Description:** EEG recordings from multiple participants watching emotionally evocative stimuli.  
- **Features:** Multiple EEG channels capturing alpha, beta, gamma, and theta activity.  
- **Labels:** Emotional states (Positive / Negative / Neutral).  

> *(You can easily substitute another dataset such as [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) for extended experiments.)*

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.10+ |
| **Data Handling** | pandas, numpy |
| **Signal Processing** | scipy.signal, mne |
| **Machine Learning** | scikit-learn, joblib |
| **Visualization** | matplotlib, seaborn, plotly |
| **Version Control** | git, GitHub |

---

## Pipeline Overview

1. **Data Ingestion**  
   Load EEG CSV data and inspect structure, sampling rate, and labels.  

2. **Preprocessing**  
   - Apply band-pass filter (1–50 Hz)  
   - Compute **Power Spectral Density (PSD)** using **Fast Fourier Transform (FFT)**  
   - Extract power values for frequency bands: delta, theta, alpha, beta, gamma  

3. **Feature Engineering**  
   Combine frequency-band features into a structured dataset ready for machine learning.  

4. **Model Training**  
   - Split dataset (80% train / 20% test)  
   - Train classifiers (Random Forest, SVM, or Logistic Regression)  
   - Evaluate with accuracy, F1-score, confusion matrix  

5. **Visualization & Reporting**  
   - Plot band power distributions across emotion classes  
   - Generate publication-quality visualizations for reports or papers  

---

## Example Results

| Model | Accuracy | F1-Score |
|--------|-----------|----------|
| RandomForestClassifier | 0.84 | 0.82 |
| SVM (RBF) | 0.80 | 0.78 |

*(Results will vary by dataset and preprocessing choices.)*

**Example Output:**
- Brainwave Power Plot – alpha vs. beta activity by emotion  
- Confusion Matrix – predicted vs. actual emotional state  
- Model Feature Importance Chart  

---

## Example Usage

```bash
# 1. Clone the repository
git clone https://github.com/neejuSingh/eeg-emotion-ai-pipeline.git
cd eeg-emotion-ai-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run preprocessing
python src/preprocess.py --input data/eeg.csv --output data/cleaned.csv

# 4. Train model
python src/train_model.py --data data/cleaned.csv --model results/model.pkl

# 5. Evaluate results
python src/evaluate.py --model results/model.pkl --data data/cleaned.csv

