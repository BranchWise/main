import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

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
        ('cat', OneHotEncoder(drop='first'), ['neighbourhood_name', 'location_type', 'age_category', 'diameter_category'])
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
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
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

# Step 7b: Select top features (e.g., top 10)
top_features = feature_importance_df.head(10)['feature']
X_train_important = X_train[top_features]
X_test_important = X_test[top_features]

# Step 7c: Retrain model with selected features
best_rf.fit(X_train_important, y_train)

# Step 7d: Evaluate with selected features
y_pred_important = best_rf.predict(X_test_important)
print("Accuracy with important features:", accuracy_score(y_test, y_pred_important))
print(classification_report(y_test, y_pred_important))

# Step 8: Prediction Function
def recommend_top_species(input_features, model, preprocessor, top_features, top_n=5):
    """
    Given input features (location data), return the top N recommended species.
    """
    input_features_transformed = preprocessor.transform(input_features)
    input_features_transformed_df = pd.DataFrame(input_features_transformed.toarray(), columns=preprocessor.get_feature_names_out())
    input_features_important = input_features_transformed_df[top_features]
    probabilities = model.predict_proba(input_features_important)
    species_prob = sorted(zip(model.classes_, probabilities[0]), key=lambda x: x[1], reverse=True)
    return [species for species, prob in species_prob[:top_n]]

# Example: Predicting top 5 species for a sample input
sample_input = features.iloc[[0]]  # Ensure sample_input has the same structure as features
top_species = recommend_top_species(sample_input, best_rf, preprocessor, top_features, top_n=5)
print("Top 5 recommended species:", top_species)