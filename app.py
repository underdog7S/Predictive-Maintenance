import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('data/predictive_maintenance.csv')

# Drop non-useful columns (adjust based on your dataset structure)
df = df.drop(['UDI', 'Product ID'], axis=1)

# Handle categorical columns (for example: 'Type' column contains 'M', 'L', 'H')
categorical_columns = ['Type']  
le = LabelEncoder()

# Encode categorical columns
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Handle missing values if necessary (drop rows or impute)
df = df.dropna()  # Example: Drop rows with missing values 
# Convert target column to binary
df['Machine failure'] = df['Machine failure'].astype(int)

# Features and target
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/model.pkl')
print("✅ Model saved to models/model.pkl")
