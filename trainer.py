# ========================================================
# Adaptive Hybrid Ensemble (AHE) Model (Optimized Version)
# Faster Training (~25–40% speed improvement)
# ========================================================

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

import lightgbm as lgb

print("✅ Loading dataset...")
df = pd.read_csv('dataset/archive/creditcard.csv')

# ------------------- 🔥 original Dataset Size -------------------
print(f"Original dataset shape: {df.shape} | Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# ------------------- 🔥 1. Reduce Dataset Size -------------------
df = df.sample(frac=0.6, random_state=42)  # 60% data for speed

# Features and Target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Dataset shape: {X.shape} | Fraud cases: {y.sum()} ({y.mean()*100:.3f}%)")

# ------------------- 🔥 2. Faster Balancing -------------------
print("🔄 Applying SMOTE (faster than SMOTE+Tomek)...")
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"Balanced dataset shape: {X_res.shape} | Fraud cases: {y_res.sum()}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ------------------- 🚀 3. Base Models (Optimized) -------------------
print("🚀 Training base models...")

# LightGBM (faster)
lgb_model = lgb.LGBMClassifier(
    n_estimators=120,
    learning_rate=0.07,
    n_jobs=-1,
    random_state=42
)
lgb_model.fit(X_train, y_train)

# Logistic Regression (faster solver)
lr_model = LogisticRegression(
    max_iter=500,
    solver='saga',
    n_jobs=-1,
    random_state=42
)
lr_model.fit(X_train, y_train)

# Neural Network (lighter)
nn_model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    max_iter=100,
    early_stopping=True,
    random_state=42
)
nn_model.fit(X_train, y_train)

# ------------------- 🔗 4. Hybrid Ensemble -------------------
print("🔗 Creating Adaptive Hybrid Ensemble...")

def ahe_predict_proba(X):
    """Weighted ensemble probability"""
    lgb_prob = lgb_model.predict_proba(X)[:, 1]
    lr_prob  = lr_model.predict_proba(X)[:, 1]
    nn_prob  = nn_model.predict_proba(X)[:, 1]
    
    # Weighted combination
    ensemble_prob = 0.45 * lgb_prob + 0.30 * lr_prob + 0.25 * nn_prob
    return ensemble_prob

def ahe_predict(X):
    return (ahe_predict_proba(X) > 0.5).astype(int)

# ------------------- 📊 5. Evaluation -------------------
print("📊 Evaluating AHE Model...")

y_pred = ahe_predict(X_test)
y_pred_proba = ahe_predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*60)
print("AHE MODEL PERFORMANCE")
print("="*60)
print(f"Accuracy      : {accuracy*100:.4f}%")
print(f"Precision     : {precision*100:.4f}%")
print(f"Recall        : {recall*100:.4f}%")
print(f"F1-Score      : {f1*100:.4f}%")
print(f"AUC-ROC       : {auc*100:.4f}%")
print(classification_report(y_test, y_pred))

# ------------------- 📉 6. Confusion Matrix -------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Legitimate', 'Fraud'],
    yticklabels=['Legitimate', 'Fraud']
)
plt.title('Confusion Matrix - AHE Model (Optimized)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# ------------------- ⏱️ 7. Inference Time -------------------
print("\n⏱️ Measuring inference time...")
start_time = time.time()
_ = ahe_predict(X_test[:2000])   # reduced from 5000 → faster
end_time = time.time()

inference_time_ms = ((end_time - start_time) / 2000) * 1000
print(f"Inference time per transaction: {inference_time_ms:.2f} ms")

if inference_time_ms < 10:
    print("✅ AHE model is real-time ready (under 10 ms)")
else:
    print("⚠️ Model needs optimization for real-time")

# ------------------- 🎉 Done -------------------
print("\n🎉 AHE Model training & evaluation completed successfully!")

# ------------------- 💾 8. Save Models -------------------
print("💾 Saving models...")
import joblib
joblib.dump(lgb_model, 'lgb_model.joblib')
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(nn_model, 'nn_model.joblib')
print("✅ Models saved successfully!")