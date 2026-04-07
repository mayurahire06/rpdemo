# trainer.py
import pandas as pd
import numpy as np
import joblib
import os
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

print("🔄 Loading Kaggle Credit Card Fraud Dataset...")
df = pd.read_csv('dataset/archive/creditcard.csv')

# Original Shape and Size
print(f"\n📊 Original Dataset Shape     : {df.shape}")
print(f"📊 Total Transactions         : {df.shape[0]:,}")
print(f"📊 Total Features (Columns)   : {df.shape[1]}")
print(f"📊 Fraud Cases                : {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
print(f"📊 Legitimate Cases           : {len(df) - df['Class'].sum()}")

# === NEW: Print Column Names and First 5 Rows ===
print("\n" + "="*80)
print("📋 COLUMN NAMES")
print("="*80)
print(list(df.columns))

print("\n🔍 FIRST 5 ROWS OF THE DATASET")
print("="*80)
print(df.head().to_string(index=False))

# Features and Target
X = df.drop('Class', axis=1)
y = df['Class']

# Balancing
print("\n🔄 Balancing with SMOTE + Tomek...")
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Train Models
print("\n🚀 Training LightGBM...")
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train)

print("🚀 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

print("🚀 Training Neural Network (MLP)...")
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                         solver='adam', max_iter=300, early_stopping=True,
                         random_state=42)
nn_model.fit(X_train, y_train)

# Save Models
os.makedirs("models", exist_ok=True)
joblib.dump(lgb_model, "models/lgb_model.pkl")
joblib.dump(lr_model, "models/lr_model.pkl")
joblib.dump(nn_model, "models/nn_model.pkl")

print("\n✅ All models saved successfully in 'models/' folder!")
print("🎉 Training completed!")