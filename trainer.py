# trainer.py

import pandas as pd
import joblib
import os

from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import lightgbm as lgb

print("🔄 Loading fraudTrain.csv for AHE Model...")

# ====================== LOAD DATA ======================
try:
    df = pd.read_csv('dataset/fraudTrain.csv')
except FileNotFoundError:
    print("❌ ERROR: fraudTrain.csv not found!")
    exit(1)

# ====================== DATASET INFO ======================
print(f"\n📊 Dataset Shape     : {df.shape}")
print(f"📊 Total Transactions: {df.shape[0]:,}")
print(f"📊 Fraud Cases       : {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.3f}%)")
print(f"📊 Legitimate Cases  : {len(df) - df['is_fraud'].sum()}")

# ====================== FEATURE SELECTION ======================
useful_cols = ['amt', 'category', 'gender', 'state', 'is_fraud']
df = df[useful_cols].copy()

# ====================== ENCODING ======================
le_category = LabelEncoder()
le_gender = LabelEncoder()
le_state = LabelEncoder()

df['category'] = le_category.fit_transform(df['category'])
df['gender'] = le_gender.fit_transform(df['gender'])
df['state'] = le_state.fit_transform(df['state'])

# ====================== SPLIT ======================
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

print("\n🔄 Applying SMOTE + Tomek...")
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# ====================== TRAIN MODELS ======================
print("\n🚀 Training Models...")

# LightGBM
# n_estimators=300 for better performance, learning_rate=0.05 for faster convergence, random_state=42 for reproducibility
#n_estimattor means number of trees, learning_rate controls how much each tree contributes to the final prediction, random_state ensures reproducibility
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Neural Network (MLP)
nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=300,
    early_stopping=True,
    random_state=42
)
nn_model.fit(X_train, y_train)

# ====================== PREDICTIONS ======================
lgb_pred = lgb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
nn_pred = nn_model.predict(X_test)

lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
lr_prob = lr_model.predict_proba(X_test)[:, 1]
nn_prob = nn_model.predict_proba(X_test)[:, 1]

# ====================== EVALUATION FUNCTION ======================
def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n🔹 {name}")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC   : {roc_auc_score(y_true, y_prob):.4f}")

# ====================== INDIVIDUAL MODEL METRICS ======================
print("\n" + "="*80)
print("📊 INDIVIDUAL MODEL PERFORMANCE")
print("="*80)

evaluate("LightGBM", y_test, lgb_pred, lgb_prob)
evaluate("Logistic Regression", y_test, lr_pred, lr_prob)
evaluate("Neural Network (MLP)", y_test, nn_pred, nn_prob)

# ====================== ENSEMBLE ======================
print("\n" + "="*80)
print("🎯 AHE ENSEMBLE PERFORMANCE")
print("="*80)

ensemble_prob = 0.45 * lgb_prob + 0.30 * lr_prob + 0.25 * nn_prob
ensemble_pred = (ensemble_prob > 0.40).astype(int)

print(f"Accuracy  : {accuracy_score(y_test, ensemble_pred):.4f}")
print(f"Precision : {precision_score(y_test, ensemble_pred):.4f}")
print(f"Recall    : {recall_score(y_test, ensemble_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, ensemble_pred):.4f}")
print(f"AUC-ROC   : {roc_auc_score(y_test, ensemble_prob):.4f}")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, ensemble_pred))

print("\n📋 Classification Report:")
print(classification_report(y_test, ensemble_pred, target_names=['Legitimate', 'Fraud']))

# ====================== SAVE MODELS ======================
os.makedirs("models", exist_ok=True)

joblib.dump(lgb_model, "models/lgb_model.pkl")
joblib.dump(lr_model, "models/lr_model.pkl")
joblib.dump(nn_model, "models/nn_model.pkl")

joblib.dump(le_category, "models/le_category.pkl")
joblib.dump(le_gender, "models/le_gender.pkl")
joblib.dump(le_state, "models/le_state.pkl")

print("\n" + "="*80)
print("✅ Training Complete & Models Saved!")
print("="*80)



















# # trainer_fixed.py
# import pandas as pd
# import joblib
# import os
# from imblearn.combine import SMOTETomek
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import LabelEncoder
# import lightgbm as lgb

# print("🔄 Loading fraudTrain.csv for AHE Model...")

# try:
#     df = pd.read_csv('dataset/fraudTrain.csv')
# except FileNotFoundError:
#     print("❌ ERROR: fraudTrain.csv not found!")
#     print("Make sure 'dataset/fraudTrain.csv' exists in the working directory.")
#     exit(1)

# # === Dataset Info ===
# print(f"\n📊 Original Dataset Shape     : {df.shape}")
# print(f"📊 Total Transactions         : {df.shape[0]:,}")
# print(f"📊 Total Features (Columns)   : {df.shape[1]}")
# print(f"📊 Fraud Cases                : {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.3f}%)")
# print(f"📊 Legitimate Cases           : {len(df) - df['is_fraud'].sum()}")

# print("\n" + "="*80)
# print("📋 COLUMN NAMES")
# print("="*80)
# print(list(df.columns))

# print("\n🔍 FIRST 5 ROWS OF THE DATASET")
# print("="*80)
# print(df.head().to_string(index=False))

# # ====================== AHE FEATURE SELECTION ======================
# useful_cols = ['amt', 'category', 'gender', 'state', 'is_fraud']
# df = df[useful_cols].copy()

# print("\n" + "="*80)
# print("🔍 UNIQUE VALUES (What Encoders Will Learn):")
# print("="*80)
# print(f"\n📌 CATEGORIES ({len(df['category'].unique())}): {sorted(df['category'].unique().tolist())}")
# print(f"📌 GENDERS ({len(df['gender'].unique())}): {sorted(df['gender'].unique().tolist())}")
# print(f"📌 STATES ({len(df['state'].unique())}): {sorted(df['state'].unique().tolist())}")

# # ====================== LABEL ENCODING ======================
# print("\n" + "="*80)
# print("🔄 ENCODING CATEGORICAL FEATURES:")
# print("="*80)

# le_category = LabelEncoder()
# le_gender = LabelEncoder()
# le_state = LabelEncoder()

# df['category'] = le_category.fit_transform(df['category'])
# df['gender'] = le_gender.fit_transform(df['gender'])
# df['state'] = le_state.fit_transform(df['state'])

# print(f"\n✅ Category Encoder Mapping:")
# for i, cat in enumerate(le_category.classes_):
#     print(f"   {cat:20} -> {i}")

# print(f"\n✅ Gender Encoder Mapping:")
# for i, gen in enumerate(le_gender.classes_):
#     print(f"   {gen:20} -> {i}")

# print(f"\n✅ State Encoder Mapping (showing first 10):")
# for i, state in enumerate(le_state.classes_[:10]):
#     print(f"   {state:20} -> {i}")
# if len(le_state.classes_) > 10:
#     print(f"   ... and {len(le_state.classes_)-10} more states")

# X = df.drop('is_fraud', axis=1)
# y = df['is_fraud']

# # ====================== BALANCING ======================
# print("\n" + "="*80)
# print("🔄 BALANCING WITH SMOTE + TOMEK:")
# print("="*80)
# print(f"Before balancing - Fraud: {(y==1).sum()}, Legitimate: {(y==0).sum()}")

# smote_tomek = SMOTETomek(random_state=42)
# X_res, y_res = smote_tomek.fit_resample(X, y)

# print(f"After balancing  - Fraud: {(y_res==1).sum()}, Legitimate: {(y_res==0).sum()}")

# X_train, X_test, y_train, y_test = train_test_split(
#     X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
# )

# print(f"\nTrain set - Fraud: {(y_train==1).sum()}, Legitimate: {(y_train==0).sum()}")
# print(f"Test set  - Fraud: {(y_test==1).sum()}, Legitimate: {(y_test==0).sum()}")

# # ====================== AHE TRAINING (3 Models) ======================
# print("\n" + "="*80)
# print("🚀 TRAINING AHE (3-MODEL ENSEMBLE):")
# print("="*80)

# print("\n1️⃣ Training LightGBM (45% weight)...")
# lgb_model = lgb.LGBMClassifier(
#     n_estimators=300, 
#     learning_rate=0.05, 
#     random_state=42,
#     verbose=-1
# )
# lgb_model.fit(X_train, y_train)
# lgb_score = lgb_model.score(X_test, y_test)
# print(f"   ✅ LGB Test Accuracy: {lgb_score:.4f}")

# print("\n2️⃣ Training Logistic Regression (30% weight)...")
# lr_model = LogisticRegression(max_iter=1000, random_state=42)
# lr_model.fit(X_train, y_train)
# lr_score = lr_model.score(X_test, y_test)
# print(f"   ✅ LR Test Accuracy: {lr_score:.4f}")

# print("\n3️⃣ Training Neural Network MLP (25% weight)...")
# nn_model = MLPClassifier(
#     hidden_layer_sizes=(128, 64), 
#     activation='relu',
#     solver='adam', 
#     max_iter=300, 
#     early_stopping=True,
#     random_state=42,
#     verbose=False
# )
# nn_model.fit(X_train, y_train)
# nn_score = nn_model.score(X_test, y_test)
# print(f"   ✅ NN Test Accuracy: {nn_score:.4f}")

# # Ensemble prediction
# from sklearn.metrics import classification_report, confusion_matrix
# lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
# lr_pred = lr_model.predict_proba(X_test)[:, 1]
# nn_pred = nn_model.predict_proba(X_test)[:, 1]
# ensemble_pred = 0.45 * lgb_pred + 0.30 * lr_pred + 0.25 * nn_pred
# ensemble_pred_class = (ensemble_pred > 0.40).astype(int)
# ensemble_score = (ensemble_pred_class == y_test).mean()
# print(f"\n🎯 AHE Ensemble Test Accuracy: {ensemble_score:.4f}")

# print("\n📊 Classification Report (Ensemble on Test Set):")
# print(classification_report(y_test, ensemble_pred_class, target_names=['Legitimate', 'Fraud']))

# # ====================== SAVE AHE MODELS & ENCODERS ======================
# os.makedirs("models", exist_ok=True)

# joblib.dump(lgb_model, "models/lgb_model.pkl")
# joblib.dump(lr_model, "models/lr_model.pkl")
# joblib.dump(nn_model, "models/nn_model.pkl")

# joblib.dump(le_category, "models/le_category.pkl")
# joblib.dump(le_gender, "models/le_gender.pkl")
# joblib.dump(le_state, "models/le_state.pkl")

# print("\n" + "="*80)
# print("✅ AHE Model Training Complete!")
# print("="*80)
# print("📁 Saved Models:")
# print("   ✓ models/lgb_model.pkl")
# print("   ✓ models/lr_model.pkl")
# print("   ✓ models/nn_model.pkl")
# print("   ✓ models/le_category.pkl")
# print("   ✓ models/le_gender.pkl")
# print("   ✓ models/le_state.pkl")
# print("\n🎉 Ready to use with app.py!")