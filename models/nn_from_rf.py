import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from datetime import datetime

# Step 1: Load Data
url = 'https://data.edmonton.ca/resource/eecg-fc54.csv'
data = pd.read_csv(url)
cleaned_data = pd.read_csv('cleaned_tree_data.csv')

# Step 2: Data Preprocessing

# 2a. Convert 'PLANTED_DATE' to tree age
data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
data['tree_age'] = (datetime.now() - data['PLANTED_DATE']).dt.days // 365  # Convert to years

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

data['condition_category'] = data['CONDITION_PERCENT'].apply(categorize_condition)

# 2c. Filter for 'Great' and 'Good' trees
suitable_trees = data[data['condition_category'].isin(['Great', 'Good'])]

# Step 3: Feature Selection
features = suitable_trees[['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'DIAMETER_BREAST_HEIGHT', 'tree_age', 'LATITUDE', 'LONGITUDE']]
target = suitable_trees['SPECIES_COMMON']

# 3a. One-hot encode categorical features
features = pd.get_dummies(features, columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE'], drop_first=True)

# 3b. Standardize numerical features
scaler = StandardScaler()
features[['DIAMETER_BREAST_HEIGHT', 'tree_age', 'LATITUDE', 'LONGITUDE']] = scaler.fit_transform(features[['DIAMETER_BREAST_HEIGHT', 'tree_age', 'LATITUDE', 'LONGITUDE']])

# 3c. Encode target labels
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 5: Build and Train MLPClassifier Model
mlp = MLPClassifier(hidden_layer_sizes=(64, 128, 256, 512, 1024), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = mlp.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 7: Prediction Function
def recommend_top_species(input_features, model, label_encoder, top_n=5):
    """
    Given input features (location data), return the top N recommended species.
    """
    probabilities = model.predict_proba(input_features)
    species_prob = sorted(zip(label_encoder.classes_, probabilities[0]), key=lambda x: x[1], reverse=True)
    return [species for species, prob in species_prob[:top_n]]

# Example: Predicting top 5 species for a sample input
sample_input = X_test[[0]]
top_species = recommend_top_species(sample_input, mlp, label_encoder, top_n=5)
print("Top 5 recommended species:", top_species)