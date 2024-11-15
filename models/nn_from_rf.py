import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from datetime import datetime

# Load Data
url = 'https://data.edmonton.ca/resource/eecg-fc54.csv'
data = pd.read_csv(url)
cleaned_data = pd.read_csv('cleaned_tree_data.csv')

# Data Preprocessing
data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
data['tree_age'] = (datetime.now() - data['PLANTED_DATE']).dt.days // 365  # Convert to years

def categorize_condition(condition):
    if condition >= 70:
        return "Great"
    elif 65 <= condition < 70:
        return "Good"
    elif condition == 65:
        return "Mediocre"
    else:
        return "Bad"

data['condition_category'] = data['CONDITION_PERCENT'].apply(categorize_condition)
suitable_trees = data[data['condition_category'].isin(['Great', 'Good'])]
features = suitable_trees[['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'DIAMETER_BREAST_HEIGHT', 'tree_age', 'LATITUDE', 'LONGITUDE']]
target = suitable_trees['SPECIES_COMMON']

# Preprocessing Pipeline
numerical_features = ['DIAMETER_BREAST_HEIGHT', 'tree_age', 'LATITUDE', 'LONGITUDE']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing to features
X = preprocessor.fit_transform(features)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(target)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning for RandomForest
rf = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=3, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_

# Hyperparameter Tuning for MLPClassifier
mlp = MLPClassifier(max_iter=500, random_state=42)
mlp_param_grid = {
    'hidden_layer_sizes': [(64, 128, 256), (128, 256, 512)],
    'alpha': [0.0001, 0.001]
}
mlp_grid_search = GridSearchCV(mlp, mlp_param_grid, cv=3, n_jobs=-1, verbose=2)
mlp_grid_search.fit(X_train, y_train)
best_mlp = mlp_grid_search.best_estimator_


estimators = [
    ('rf', best_rf),
    ('mlp', best_mlp)
]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(random_state=42))
stacking_clf.fit(X_train, y_train)


y_pred = stacking_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


def recommend_top_species(input_features, model, label_encoder, top_n=5):
    """
    Given input features (location data), return the top N recommended species.
    """
    probabilities = model.predict_proba(input_features)
    species_prob = sorted(zip(label_encoder.classes_, probabilities[0]), key=lambda x: x[1], reverse=True)
    return [species for species, prob in species_prob[:top_n]]

# Example: Predicting top 5 species for a sample input
sample_input = X_test[[0]]
top_species = recommend_top_species(sample_input, stacking_clf, label_encoder, top_n=5)
print("Top 5 recommended species:", top_species)