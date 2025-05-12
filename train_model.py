import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data/predictive_maintenance.csv')

# Print columns to confirm
print("📋 Columns:", df.columns)

# Drop non-numeric or identifier columns
df = df.drop(['UDI', 'Product ID'], axis=1)

# Encode 'Type' column (categorical to numeric)
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

# Check the unique encoded values of 'Type'
print("Encoded 'Type' values:", df['Type'].unique())

# Check if any columns are still non-numeric
print("🔍 Checking for non-numeric values in the dataframe...")
print(df.dtypes)

# Ensure all columns are numeric (except 'Machine failure')
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]) and col != 'Machine failure':
        print(f"⚠️ Non-numeric column found: {col}")
    else:
        print(f"✅ {col} is numeric")

# Features and target
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with class_weight balanced
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Save model and feature names
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
print("✅ Model saved to models/model.pkl")
print("✅ Feature names saved to models/feature_names.pkl")
