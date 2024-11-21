# File path: hybrid_nn_rf_model.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from datetime import datetime
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Custom transformer to use MLPClassifier for feature extraction
class NeuralNetFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_layer_sizes=(64, 32), max_iter=500, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.nn = None  # Neural network model

    def fit(self, X, y=None):
        self.nn = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.nn.fit(X, y)
        return self

    def transform(self, X):
        return self.nn.predict_proba(X)

# Step 1: Load and Preprocess Data
url = 'https://data.edmonton.ca/resource/eecg-fc54.csv'
data = pd.read_csv(url)

# Convert columns to lowercase
data.columns = data.columns.str.lower()

# Data preprocessing
data['planted_date'] = pd.to_datetime(data['planted_date'], errors='coerce')
data['tree_age'] = (datetime.now() - data['planted_date']).dt.days // 365  # Convert to years

def categorize_condition(condition):
    if condition >= 70:
        return "Great"
    elif 65 <= condition < 70:
        return "Good"
    elif condition == 65:
        return "Mediocre"
    else:
        return "Bad"

data['condition_category'] = data['condition_percent'].apply(categorize_condition)
suitable_trees = data[data['condition_category'].isin(['Great', 'Good'])].copy()

suitable_trees['age_category'] = pd.cut(
    suitable_trees['tree_age'], bins=[0, 10, 20, 50, 100], labels=['0-10', '10-20', '20-50', '50-100']
)
suitable_trees['diameter_category'] = pd.cut(
    suitable_trees['diameter_breast_height'], bins=[0, 10, 20, 50, 100], labels=['0-10', '10-20', '20-50', '50-100']
)

# Feature selection and encoding
features = suitable_trees[['neighbourhood_name', 'location_type', 'diameter_breast_height', 'tree_age', 'latitude', 'longitude', 'age_category', 'diameter_category']]
target = suitable_trees['species']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['neighbourhood_name', 'location_type', 'age_category', 'diameter_category']),
        ('num', StandardScaler(), ['diameter_breast_height', 'tree_age', 'latitude', 'longitude'])
    ],
    remainder='passthrough'
)

# Step 2: Split Data
X_transformed = preprocessor.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, target, test_size=0.2, random_state=42)

# Step 3: Hybrid Pipeline
hybrid_model = Pipeline([
    ('nn_features', NeuralNetFeatureExtractor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the hybrid model
hybrid_model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = hybrid_model.predict(X_test)
y_proba = hybrid_model.predict_proba(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# Ensure y_test is binarized correctly
unique_train_classes = np.unique(y_train)
unique_test_classes = np.unique(y_test)

# Binarize y_test to match the structure of y_proba
y_test_binarized = label_binarize(y_test, classes=unique_train_classes)

# Check the shape of y_proba
print("Shape of y_proba:", y_proba.shape)
print("Shape of y_test_binarized:", y_test_binarized.shape)

# Align the predicted probabilities with y_test classes
class_indices = [np.where(unique_train_classes == cls)[0][0] for cls in unique_test_classes if cls in unique_train_classes]
aligned_y_proba = y_proba[:, class_indices]

# Filter out classes with only one unique value in y_test_binarized
valid_classes = [i for i in range(y_test_binarized.shape[1]) if len(np.unique(y_test_binarized[:, i])) > 1]
filtered_y_test_binarized = y_test_binarized[:, valid_classes]

# Ensure valid_classes are within the bounds of aligned_y_proba
valid_class_indices = [i for i in valid_classes if i < aligned_y_proba.shape[1]]
filtered_aligned_y_proba = aligned_y_proba[:, valid_class_indices]

# Calculate ROC AUC score
roc_auc = roc_auc_score(filtered_y_test_binarized, filtered_aligned_y_proba, multi_class='ovr')

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
classification_rep = classification_report(y_test, y_pred)

# Results
print("Hybrid Model Evaluation:")
print("Accuracy:", accuracy)
print("F1 Score (weighted):", f1_weighted)
print("ROC AUC Score:", roc_auc)
print("\nClassification Report:\n", classification_rep)