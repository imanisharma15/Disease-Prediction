# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2. Load Dataset
data = pd.read_csv("clean_disease_dataset.csv", sep="\t")

print("\n===== DATASET PREVIEW =====")
print(data.head())

print("\n===== DATASET SHAPE =====")
print(data.shape)

print("\n===== COLUMN NAMES =====")
print(data.columns)

# 3. Data Preprocessing
data.columns = data.columns.str.strip()

X = data.drop("Disease", axis=1)
y = data["Disease"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# 4. Scaling (for SVM & Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)


# 6. Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)


# 7. SVM
svm_model = SVC(kernel='rbf', C=10)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)


# 8. Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)


# 9. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)


# 10. Accuracy
print("\n===== ACCURACY =====")

dt_acc = accuracy_score(y_test, dt_pred)
nb_acc = accuracy_score(y_test, nb_pred)
svm_acc = accuracy_score(y_test, svm_pred)
lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_acc)
print("Naive Bayes Accuracy:", nb_acc)
print("SVM Accuracy:", svm_acc)
print("Logistic Regression Accuracy:", lr_acc)
print("Random Forest Accuracy:", rf_acc)


# 11. Confusion Matrix
print("\n===== CONFUSION MATRIX =====")

print("\nDecision Tree:\n", confusion_matrix(y_test, dt_pred))
print("\nNaive Bayes:\n", confusion_matrix(y_test, nb_pred))
print("\nSVM:\n", confusion_matrix(y_test, svm_pred))
print("\nLogistic Regression:\n", confusion_matrix(y_test, lr_pred))
print("\nRandom Forest:\n", confusion_matrix(y_test, rf_pred))


# 12. Classification Report
print("\n===== CLASSIFICATION REPORT =====")

print("\nDecision Tree:\n", classification_report(y_test, dt_pred))
print("\nNaive Bayes:\n", classification_report(y_test, nb_pred))
print("\nSVM:\n", classification_report(y_test, svm_pred))
print("\nLogistic Regression:\n", classification_report(y_test, lr_pred))
print("\nRandom Forest:\n", classification_report(y_test, rf_pred))


# 13. Sample Prediction (Using Best Model: Random Forest)
print("\n===== SAMPLE PREDICTION =====")

sample = X_test.iloc[0].values.reshape(1, -1)

prediction = rf_model.predict(sample)
predicted_disease = le.inverse_transform(prediction)

print("Predicted Disease:", predicted_disease[0])


# 14. Graph
models = ['Decision Tree', 'Naive Bayes', 'SVM', 'Logistic Regression', 'Random Forest']
accuracies = [dt_acc, nb_acc, svm_acc, lr_acc, rf_acc]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()