from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import os

app = Flask(__name__)

# Load dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path, sep='\t')

# Replace invalid zero values with NaN
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns_to_replace:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].mean())

# Features and Target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

accuracies = {}

# Train each model
for name, model in models.items():
    if name in ['SVM', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    accuracies[name] = accuracy_score(y_test, pred)

# Save SVM model and scaler for prediction
pickle.dump(models['SVM'], open('svm_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Create static folder if not exists
if not os.path.exists('static'):
    os.makedirs('static')

# 1. Accuracy Graph
plt.figure(figsize=(10,6))
plt.bar(accuracies.keys(), accuracies.values())
plt.title('Model Accuracy Comparison')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.xticks(rotation=15)
for i, acc in enumerate(accuracies.values()):
    plt.text(i, acc + 0.01, str(round(acc, 2)), ha='center')
plt.tight_layout()
plt.savefig('static/accuracy_graph.png')
plt.close()

# 2. Confusion Matrix for SVM
svm_model = models['SVM']
svm_pred = svm_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('static/confusion_matrix.png')
plt.close()

# 3. Feature Importance Graph for Random Forest
rf_model = models['Random Forest']
importance = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
plt.bar(features, importance)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()

# 4. Pie Chart of Diabetic vs Non-Diabetic
outcome_counts = data['Outcome'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(outcome_counts, labels=['Non-Diabetic', 'Diabetic'], autopct='%1.1f%%')
plt.title('Diabetic vs Non-Diabetic Distribution')
plt.savefig('static/pie_chart.png')
plt.close()

# 5. ROC Curve for SVM
svm_probs = svm_model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, svm_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.legend(loc='lower right')
plt.savefig('static/roc_curve.png')
plt.close()

# Load trained model
model = pickle.load(open('svm_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = pd.DataFrame({
            'Pregnancies': [float(request.form['pregnancies'])],
            'Glucose': [float(request.form['glucose'])],
            'BloodPressure': [float(request.form['blood_pressure'])],
            'SkinThickness': [float(request.form['skin_thickness'])],
            'Insulin': [float(request.form['insulin'])],
            'BMI': [float(request.form['bmi'])],
            'DiabetesPedigreeFunction': [float(request.form['diabetes_pedigree'])],
            'Age': [float(request.form['age'])]
        })

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        if prediction[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'

        return render_template('index.html', prediction_text=f'Prediction Result: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)