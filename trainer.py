# trainer.py
import pandas as pd
import joblib
import os
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

print("🔄 Loading fraudTrain.csv for AHE Model...")

df = pd.read_csv('dataset/fraudTrain.csv')

# === Dataset Info (as you requested) ===
print(f"\n📊 Original Dataset Shape     : {df.shape}")
print(f"📊 Total Transactions         : {df.shape[0]:,}")
print(f"📊 Total Features (Columns)   : {df.shape[1]}")
print(f"📊 Fraud Cases                : {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.3f}%)")
print(f"📊 Legitimate Cases           : {len(df) - df['is_fraud'].sum()}")

print("\n" + "="*80)
print("📋 COLUMN NAMES")
print("="*80)
print(list(df.columns))

print("\n🔍 FIRST 5 ROWS OF THE DATASET")
print("="*80)
print(df.head().to_string(index=False))

# ====================== AHE FEATURE SELECTION ======================
useful_cols = ['amt', 'category', 'gender', 'state', 'is_fraud']
df = df[useful_cols].copy()

# ====================== LABEL ENCODING ======================
le_category = LabelEncoder()
le_gender = LabelEncoder()
le_state = LabelEncoder()

df['category'] = le_category.fit_transform(df['category'])
df['gender'] = le_gender.fit_transform(df['gender'])
df['state'] = le_state.fit_transform(df['state'])

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# ====================== BALANCING ======================
print("\n🔄 Balancing with SMOTE + Tomek...")
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ====================== AHE TRAINING (3 Models) ======================
print("\n🚀 Training LightGBM (Part of AHE)...")
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train)

print("🚀 Training Logistic Regression (Part of AHE)...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

print("🚀 Training Neural Network MLP (Part of AHE)...")
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                         solver='adam', max_iter=300, early_stopping=True,
                         random_state=42)
nn_model.fit(X_train, y_train)

# ====================== SAVE AHE MODELS & ENCODERS ======================
os.makedirs("models", exist_ok=True)

joblib.dump(lgb_model, "models/lgb_model.pkl")
joblib.dump(lr_model, "models/lr_model.pkl")
joblib.dump(nn_model, "models/nn_model.pkl")

joblib.dump(le_category, "models/le_category.pkl")
joblib.dump(le_gender, "models/le_gender.pkl")
joblib.dump(le_state, "models/le_state.pkl")

print("\n✅ AHE Model (3 base models + encoders) saved successfully!")
print("🎉 Adaptive Hybrid Ensemble Training Completed!")