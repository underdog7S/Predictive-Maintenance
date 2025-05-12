# 🔧 Predictive Maintenance - Machine Failure Prediction

This project leverages machine learning to predict machine failure based on sensor data from machines. The goal is to classify whether a machine will fail based on various sensor readings and characteristics, enabling proactive maintenance actions.

---

## 📚 Table of Contents

- [Project Description]  
- [Technologies Used]
- [Getting Started] 
- [Data Preprocessing]  
- [Model Training]
- [Model Evaluation]
- [How to Use]
- [License]

---

## 📄 Project Description

The project uses a dataset containing multiple sensor readings from industrial machines and their associated failure status. 
The primary objective is to predict machine failure based on sensor inputs to enable **predictive maintenance**, reduce downtime,
 and improve operational efficiency.

---

## 🛠️ Technologies Used

- **Python** – Programming language for development  
- **Pandas** – Data manipulation and preprocessing  
- **NumPy** – Numerical operations  
- **Scikit-learn** – Machine learning model development  
- **Matplotlib** – Confusion matrix visualization  
- **Joblib** – Model serialization  
- **RandomForestClassifier** – Classification model used for training  

---



### ✅ Prerequisites

Ensure you have Python 3.12 installed along with the following packages:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
📁 Dataset
Place your dataset file predictive_maintenance.csv in the data/ directory. The dataset must include:

UDI

Product ID

Type

Sensor values

Machine failure flag

⚙️ Data Preprocessing
Steps involved:

Drop non-informative columns: UDI and Product ID

Encode categorical feature Type using LabelEncoder

Convert target variable Machine failure to integer

Split the data into features (X) and target (y)

🎯 Model Training
We use a Random Forest Classifier with class_weight='balanced' to handle imbalanced data.

python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
📈 Model Evaluation
Model performance is evaluated using:

Classification Report – Precision, recall, F1-score

Confusion Matrix – Visual representation of prediction vs. actual

Accuracy Score – Overall performance metric

python
Copy
Edit
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
The confusion matrix is visualized using matplotlib.

🧪 How to Use
Ensure predictive_maintenance.csv is in the data/ folder

Run the training script

The trained model is saved to models/model.pkl

Feature names are saved as models/feature_names.pkl

To make predictions later:

python
Copy
Edit
import joblib

model = joblib.load('models/model.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Make predictions
predictions = model.predict(X_test)
📄 License
This project is licensed under the MIT License.

👨‍💻 Author
Shadab Sheikh