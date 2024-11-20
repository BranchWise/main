import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Load Data
url = 'https://data.edmonton.ca/resource/eecg-fc54.csv'
data = pd.read_csv(url)
cleaned_data = pd.read_csv('data/preprocessing/cleaned_tree_data.csv')

# Data Preprocessing
data['PLANTED_DATE'] = pd.to_datetime(data['planted_date'], errors='coerce')
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

data['condition_category'] = data['condition_percent'].apply(categorize_condition)
suitable_trees = data[data['condition_category'].isin(['Great', 'Good'])]
features = suitable_trees[['neighbourhood_name', 'location_type', 'diameter_breast_height', 'tree_age', 'latitude', 'longitude']]
target = suitable_trees['species']

# Encode target labels
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

# Preprocessing Pipeline
numerical_features = ['diameter_breast_height', 'tree_age', 'latitude', 'longitude']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['neighbourhood_name', 'location_type']
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
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Define the model
rf = RandomForestClassifier()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Perform grid search
rf_grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train_preprocessed, y_train)

# Make predictions
y_pred = rf_grid_search.predict(X_test_preprocessed)

# Print classification report
print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=label_encoder.classes_))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")