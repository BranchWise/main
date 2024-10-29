import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Step 1: Load Data
file_path = '/Users/sophiecabungcal/Downloads/Trees_20241025.csv'  # Update this path to your local CSV file
data = pd.read_csv(file_path)

# Step 2: Data Preprocessing
# 2a. Handle missing values
data['CONDITION_PERCENT'] = data['CONDITION_PERCENT'].fillna(data['CONDITION_PERCENT'].mean())
data['DIAMETER_BREAST_HEIGHT'] = data['DIAMETER_BREAST_HEIGHT'].fillna(data['DIAMETER_BREAST_HEIGHT'].mean())
data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
data['PLANTED_DATE'] = data['PLANTED_DATE'].fillna(datetime.now())

# 2b. Convert 'PLANTED_DATE' to tree age
data['tree_age'] = (datetime.now() - data['PLANTED_DATE']).dt.days // 365  # Convert to years

# 2c. Create tree condition labels
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

# 2d. Filter for 'Great' and 'Good' trees
suitable_trees = data[data['condition_category'].isin(['Great', 'Good'])]

# Filter out classes with fewer than 2 members
class_counts = suitable_trees['SPECIES_COMMON'].value_counts()
suitable_trees = suitable_trees[suitable_trees['SPECIES_COMMON'].isin(class_counts[class_counts >= 2].index)]

# Step 3: Feature Selection
features = suitable_trees[['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'DIAMETER_BREAST_HEIGHT', 'tree_age', 'LATITUDE', 'LONGITUDE']]
target = suitable_trees['SPECIES_COMMON']

# 3a. One-hot encode categorical features
features = pd.get_dummies(features, columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE'], drop_first=True)

# Step 4: Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Step 5: Model Training
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 7: Feature Importance
feature_importances = rf.feature_importances_
feature_names = features.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances:\n", importance_df)

# Step 8: Prediction Function
def recommend_top_species(input_features, model, top_n=5):
    """
    Given input features (location data), return the top N recommended species.
    """
    probabilities = model.predict_proba(input_features)
    species_prob = sorted(zip(model.classes_, probabilities[0]), key=lambda x: x[1], reverse=True)
    return [species for species, prob in species_prob[:top_n]]

# Example: Predicting top 5 species for a sample input
sample_input = X_test.iloc[[0]]
top_species = recommend_top_species(sample_input, rf, top_n=5)
print("Top 5 recommended species:", top_species)