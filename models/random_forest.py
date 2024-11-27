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

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features_transformed_df, target, test_size=0.2, random_state=42)

# Step 6: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the model with the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = best_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
# Check unique classes in y_train and y_test
print("Unique classes in y_train:", np.unique(y_train))
print("Unique classes in y_test:", np.unique(y_test))

# Check the shape of the predicted probabilities
print("Shape of predicted probabilities:", best_rf.predict_proba(X_test).shape)
print(classification_report(y_test, y_pred))

# Step 7a: Extract feature importances
importances = best_rf.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for visualization and selection
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("Feature Importances:\n", feature_importance_df)

# Ensure the columns of X_test match those of X_train
X_test = X_test[X_train.columns]

# Step 7b: Select top features (e.g., top 10)
top_features = feature_importance_df.head(10)['feature']
X_train_important = X_train[top_features]
X_test_important = X_test[top_features]

# Step 7c: Retrain model with selected features
best_rf.fit(X_train_important, y_train)

# Step 7d: Evaluate with selected features
y_pred_important = best_rf.predict(X_test_important)
y_pred_proba_important = best_rf.predict_proba(X_test_important)

# Align the predicted probabilities with y_test classes
unique_test_classes = np.unique(y_test)

# Find the indices of test classes in model classes
class_indices = [np.where(best_rf.classes_ == cls)[0][0] for cls in unique_test_classes if cls in best_rf.classes_]

# Subset predict_proba to only include columns for the test classes
aligned_pred_proba = y_pred_proba_important[:, class_indices]

# Ensure y_test has the same classes as y_train
classes = np.unique(y_train)
y_test_binarized = label_binarize(y_test, classes=classes)

# Check if there is more than one class in y_test
if len(np.unique(y_test)) > 1:
    roc_auc = roc_auc_score(y_test_binarized, aligned_pred_proba, multi_class='ovr')
    print(f"ROC AUC Score: {roc_auc}")
else:
    print("ROC AUC score is not defined for a single class.")

# Standard metrics
print("Accuracy with important features:", accuracy_score(y_test, y_pred_important))
print("F1 Score with important features:", f1_score(y_test, y_pred_important, average='weighted'))
print(classification_report(y_test, y_pred_important))