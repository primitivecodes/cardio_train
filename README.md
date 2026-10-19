# Cardiovascular Disease Prediction: Traditional ML vs. Deep Learning

Binary classification of cardiovascular disease from non-invasive patient data, comparing scikit-learn models with a TensorFlow/Keras neural network.

## Project Links

| Resource | Link |
|---|---|
| Written report (PDF) | `<ADD LINK>` |
| Google Drive (raw dataset zip + saved models) | `<ADD LINK>` (access: Anyone with the link can view) |
| Presentation video (5-10 min) | `<ADD LINK>` |

## Problem Statement

Cardiovascular disease is a leading cause of death worldwide. This project asks whether a patient's cardiovascular disease status can be predicted from routine, non-invasive measurements (demographics, blood pressure, body measurements, lifestyle), and whether deep learning offers any advantage over tuned traditional models on this tabular data.

## Dataset

- **Source:** [Cardiovascular Disease dataset (Kaggle, `sulianova/cardiovascular-disease-dataset`)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Size:** 70,000 rows x 13 columns (68,563 rows after cleaning)
- **Target:** `cardio` (0 = no disease, 1 = disease), approximately balanced
- **Features:** age (days), gender, height, weight, systolic/diastolic blood pressure (`ap_hi`, `ap_lo`), cholesterol, glucose, smoking, alcohol, physical activity
- **Note:** the raw dataset is **not** committed to this repository. It is downloaded automatically by the notebook via `kagglehub` and also provided as a zip in the Google Drive folder above.

## Pipeline Overview

1. **Cleaning:** removed 1,437 rows with physiologically implausible blood pressure, height or weight values; dropped `id`.
2. **Feature engineering:** `age_years`, `bmi`, `pulse_pressure`.
3. **Splitting:** stratified 70/10/20 train/validation/test split (47,993 / 6,857 / 13,713), `random_state=42`.
4. **Scaling:** `StandardScaler` fit on the training set only.
5. **Traditional ML (scikit-learn):** Logistic Regression, Random Forest, Gradient Boosting, SVM; `RandomizedSearchCV` tuning of Random Forest.
6. **Deep learning (TensorFlow/Keras):** Sequential ANN (128-64-32, BatchNorm, Dropout), `tf.data` input pipeline, EarlyStopping and ModelCheckpoint callbacks.
7. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, ROC curves, confusion matrices, training curves (ANN) and learning curves (Gradient Boosting).

## Results (Test Set, n = 13,713)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Deep Learning (ANN) | 0.7341 | 0.7487 | 0.6962 | 0.7215 | 0.7965 |
| Gradient Boosting | 0.7311 | 0.7483 | 0.6879 | 0.7168 | 0.7959 |
| Tuned Random Forest | 0.7308 | 0.7520 | 0.6801 | 0.7142 | 0.7950 |
| Logistic Regression | 0.7235 | 0.7491 | 0.6632 | 0.7035 | 0.7860 |
| SVM (5k-sample subset) | 0.7265 | 0.7518 | 0.6675 | 0.7071 | 0.7807 |
| Random Forest (baseline) | 0.7125 | 0.7165 | 0.6930 | 0.7045 | 0.7705 |

All models plateau at roughly 73% accuracy / 0.80 AUC. Differences between the top three models are very small. See the report for discussion of dataset limitations and error analysis.

## Repository Structure

```
.
├── Cardio_Predictive_Modeling_Research.ipynb   # Full pipeline (run top to bottom)
├── requirements.txt                            # Python dependencies
└── README.md
```

## Getting Started

### Option A: Google Colab (recommended)
1. Open the notebook in Colab (upload it or use "Open in Colab" from GitHub).
2. Run all cells in order. The dataset is fetched automatically with `kagglehub`.

### Option B: Local
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
`kagglehub` may require Kaggle credentials. See the [kagglehub docs](https://github.com/Kaggle/kagglehub) if the download fails.

## Saved Models

Saved models are stored in the Google Drive folder (not in this repo):

| File | Description |
|---|---|
| `best_traditional_model.pkl` | Tuned Random Forest (joblib) |
| `best_deep_learning_model.h5` | Keras ANN (best weights restored by early stopping) |

Loading examples:
```python
import joblib
from tensorflow.keras.models import load_model

rf = joblib.load("best_traditional_model.pkl")
ann = load_model("best_deep_learning_model.h5")
```
Inputs must be preprocessed exactly as in the notebook (same cleaning, engineered features and column order; scaled features for the ANN).

## Reproducibility

- Random seed 42 is used for splits, models and tuning. Minor run-to-run differences in the ANN are possible due to TensorFlow nondeterminism.
- Tested with Python 3.13 and TensorFlow 2.21 in Google Colab.

## Limitations

- Features are non-invasive only (no ECG or lab results), which limits achievable accuracy.
- Labels and measurements are self-reported or noisy in places; cleaning rules are heuristic.
- The SVM was trained on a 5,000-sample subset for speed.
- Only Random Forest received formal hyperparameter search in the base notebook.

## Author

`<Your Name>` | `<Course / Institution>` | `<Date>`

## Acknowledgements

Dataset by Svetlana Ulianova (Kaggle). Built with scikit-learn, TensorFlow/Keras, pandas, NumPy, matplotlib and seaborn.
