import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the trained model
rf = joblib.load('random_forest_model.pkl')

# Load the data
file_path = '/Users/sophiecabungcal/Downloads/Trees_20241025.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
data['CONDITION_PERCENT'] = data['CONDITION_PERCENT'].fillna(data['CONDITION_PERCENT'].mean())
data['DIAMETER_BREAST_HEIGHT'] = data['DIAMETER_BREAST_HEIGHT'].fillna(data['DIAMETER_BREAST_HEIGHT'].mean())
data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
data['PLANTED_DATE'] = data['PLANTED_DATE'].fillna(datetime.now())
data['tree_age'] = (datetime.now() - data['PLANTED_DATE']).dt.days // 365  # Convert to years
data['NEIGHBOURHOOD_NAME'] = data['NEIGHBOURHOOD_NAME'].str.lower()  # Normalize to lowercase
data['LOCATION_TYPE'] = data['LOCATION_TYPE'].str.lower()  # Normalize to lowercase

def categorize_condition(condition):
    """
    Defines tree condition labels
    """
    if condition >= 70:
        return "Great"
    elif 65 <= condition < 70:
        return "Good"
    elif condition == 65:
        return "Mediocre"
    else:
        return "Bad"

data['condition_category'] = data['CONDITION_PERCENT'].apply(categorize_condition)

# Filter for 'Great' and 'Good' trees
suitable_trees = data[data['condition_category'].isin(['Great', 'Good'])]

# Filter out classes w/ <2 members
class_counts = suitable_trees['SPECIES_COMMON'].value_counts()
suitable_trees = suitable_trees[suitable_trees['SPECIES_COMMON'].isin(class_counts[class_counts >= 2].index)]

# Feature Selection
features = suitable_trees[['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'DIAMETER_BREAST_HEIGHT', 'tree_age', 'LATITUDE', 'LONGITUDE']]
target = suitable_trees['SPECIES_COMMON']

# One-hot encode categorical features
features = pd.get_dummies(features, columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE'], drop_first=True)

# Train-Test Split w/ Stratification
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model (metrics)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')
log_loss_value = log_loss(y_test, rf.predict_proba(X_test))

print(f"Accuracy: {accuracy}")  # Proportion of correct predictions.
print(f"Precision: {precision}")  # Proportion of true positive predictions.
print(f"Recall: {recall}")  # Proportion of actual positive instances that were correctly predicted.
print(f"F1-Score: {f1}")  # Harmonic mean of precision and recall.
print(f"Confusion Matrix:\n{conf_matrix}")  # Table showing the actual versus predicted classifications.
print(f"ROC-AUC: {roc_auc}")  # Area under the ROC curve; higher values are better.
print(f"Log Loss: {log_loss_value}")  # Measure of uncertainty; lower values are better.

# Visualize the confusion matrix 
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()