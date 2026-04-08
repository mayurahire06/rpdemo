import pandas as pd

# Check your actual dataset
df = pd.read_csv('dataset/fraudTrain.csv')

print("=" * 80)
print("🔍 UNIQUE CATEGORIES (What your encoders expect):")
print("=" * 80)
print(df['category'].unique())
print(f"\nTotal unique categories: {len(df['category'].unique())}")

print("\n" + "=" * 80)
print("🔍 UNIQUE STATES (What your encoders expect):")
print("=" * 80)
print(sorted(df['state'].unique()))
print(f"\nTotal unique states: {len(df['state'].unique())}")

print("\n" + "=" * 80)
print("🔍 UNIQUE GENDERS:")
print("=" * 80)
print(df['gender'].unique())

print("\n" + "=" * 80)
print("📊 SAMPLE FRAUD vs LEGITIMATE TRANSACTIONS:")
print("=" * 80)
print("\n✅ LEGITIMATE (is_fraud = 0):")
print(df[df['is_fraud'] == 0][['amt', 'category', 'gender', 'state', 'is_fraud']].head(10))

print("\n🚨 FRAUDULENT (is_fraud = 1):")
print(df[df['is_fraud'] == 1][['amt', 'category', 'gender', 'state', 'is_fraud']].head(10))

print("\n" + "=" * 80)
print("💰 AMOUNT STATISTICS:")
print("=" * 80)
print(f"Legitimate - Min: ${df[df['is_fraud']==0]['amt'].min():.2f}, Max: ${df[df['is_fraud']==0]['amt'].max():.2f}, Mean: ${df[df['is_fraud']==0]['amt'].mean():.2f}")
print(f"Fraudulent - Min: ${df[df['is_fraud']==1]['amt'].min():.2f}, Max: ${df[df['is_fraud']==1]['amt'].max():.2f}, Mean: ${df[df['is_fraud']==1]['amt'].mean():.2f}")