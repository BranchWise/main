import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Step 1: Load Data
url = 'https://data.edmonton.ca/resource/eecg-fc54.csv'
data = pd.read_csv(url)

# Convert columns to lowercase
data.columns = data.columns.str.lower()

# Step 2: Data Preprocessing

# 2a. Convert 'planted_date' to tree age
data['planted_date'] = pd.to_datetime(data['planted_date'], errors='coerce')
data['tree_age'] = (datetime.now() - data['planted_date']).dt.days // 365  # Convert to years

# 2b. Create tree condition labels
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

# 2c. Filter for 'Great' and 'Good' trees
suitable_trees = data[data['condition_category'].isin(['Great', 'Good'])]

# Step 3: Feature Engineering
suitable_trees['age_category'] = pd.cut(suitable_trees['tree_age'], bins=[0, 10, 20, 50, 100], labels=['0-10', '10-20', '20-50', '50-100'])
suitable_trees['diameter_category'] = pd.cut(suitable_trees['diameter_breast_height'], bins=[0, 10, 20, 50, 100], labels=['0-10', '10-20', '20-50', '50-100'])

# Step 4: Feature Selection
features = suitable_trees[['neighbourhood_name', 'location_type', 'diameter_breast_height', 'tree_age', 'latitude', 'longitude', 'age_category', 'diameter_category']]
target = suitable_trees['species']

# 4a. One-hot encode categorical features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['neighbourhood_name', 'location_type', 'age_category', 'diameter_category']),
        ('num', StandardScaler(), ['diameter_breast_height', 'tree_age', 'latitude', 'longitude'])
    ],
    remainder='passthrough'
)

# Apply the transformations to the features
features_transformed = preprocessor.fit_transform(features)

# Convert the sparse matrix to a dense DataFrame
features_transformed_df = pd.DataFrame(features_transformed.toarray(), columns=preprocessor.get_feature_names_out())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features_transformed_df, target, test_size=0.2, random_state=42
)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False],
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best Parameters
best_rf = grid_search.best_estimator_

# Train Model
best_rf.fit(X_train, y_train)

# Evaluation Metrics
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("Training F1 Score:", f1_score(y_train, y_train_pred, average='weighted'))
print("Testing F1 Score:", f1_score(y_test, y_test_pred, average='weighted'))

print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))

# Ensure predicted probabilities align with `y_test`
if len(np.unique(y_test)) > 1:
    # Align `predict_proba` columns with `y_test` classes
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
    y_test_proba = best_rf.predict_proba(X_test)
    
    # Align predicted probabilities for test classes
    test_classes = np.unique(y_test)
    aligned_proba = np.zeros((y_test_proba.shape[0], len(test_classes)))
    for i, cls in enumerate(test_classes):
        aligned_proba[:, i] = y_test_proba[:, np.where(best_rf.classes_ == cls)[0][0]]
    
    # Compute ROC AUC Score
    roc_auc = roc_auc_score(y_test_binarized, aligned_proba, multi_class='ovr')
    print(f"ROC AUC Score: {roc_auc}")
else:
    print("ROC AUC score is not defined for single-class testing data.")

# Feature Importances
importances = best_rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nFeature Importances:\n", feature_importance_df)

# Select Top Features
top_features = feature_importance_df.head(10)['feature']
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Retrain and Evaluate with Top Features
best_rf.fit(X_train_top, y_train)
y_test_top_pred = best_rf.predict(X_test_top)

print("\nEvaluation with Top Features:")
print("Testing Accuracy:", accuracy_score(y_test, y_test_top_pred))
print("Testing F1 Score:", f1_score(y_test, y_test_top_pred, average='weighted'))
print("\nClassification Report (Top Features):\n", classification_report(y_test, y_test_top_pred))
