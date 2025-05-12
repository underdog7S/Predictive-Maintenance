import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
df = pd.read_csv('data/predictive_maintenance.csv')

# Drop non-useful or string-based columns
df = df.drop(['UDI', 'Product ID'], axis=1)

# Encode 'Type' (convert categories to numbers)
df['Type'] = df['Type'].astype('category').cat.codes

# Target column
df['Machine failure'] = df['Machine failure'].astype(int)

# Features and target
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("✅ Model saved to models/model.pkl")
