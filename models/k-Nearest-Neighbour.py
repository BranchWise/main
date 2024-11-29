import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

# Load the modified data with formatted species names
data = pd.read_csv('Trees_Modified_Species_Common.csv')

# Step 1: Data Preprocessing

# 1a. Convert 'PLANTED_DATE' to tree age
data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
data['tree_age'] = (datetime.now() - data['PLANTED_DATE']).dt.days // 365  # Convert to years

# 1b. Categorize tree condition
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
suitable_trees = data[data['condition_category'].isin(['Great', 'Good'])].copy()

# Step 2: Feature Selection and Encoding
le_location = LabelEncoder()
le_neighbourhood = LabelEncoder()

suitable_trees['LOCATION_TYPE'] = le_location.fit_transform(suitable_trees['LOCATION_TYPE'])
suitable_trees['NEIGHBOURHOOD_NAME'] = le_neighbourhood.fit_transform(suitable_trees['NEIGHBOURHOOD_NAME'])

X = suitable_trees[['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'CONDITION_PERCENT', 'DIAMETER_BREAST_HEIGHT', 'tree_age']]
y = suitable_trees['SPECIES_COMMON']

# Handle missing values
X = X.dropna()
y = y.loc[X.index]

class_counts = y.value_counts()
valid_classes = class_counts[class_counts > 1].index  # Classes with more than 1 sample
X_filtered = X[y.isin(valid_classes)]
y_filtered = y[y.isin(valid_classes)]

# Now perform stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)
# Step 3: Train-Test Split

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train k-Nearest Neighbors Model
param_grid = {'n_neighbors': list(range(1, 10, 1))}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=2)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
print(f"Optimal n_neighbors found: {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Errors
train_error = 1 - accuracy_score(y_train, y_train_pred)
test_error = 1 - accuracy_score(y_test, y_test_pred)

print(f"Training Error: {train_error:.4f}")
print(f"Testing Error: {test_error:.4f}")

# Step 6: Model Evaluation
y_pred = knn.predict(X_test)  
print(classification_report(y_test, y_pred, zero_division=0))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # Weighted for multiclass
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\nSummary of Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1 Score (Weighted): {f1:.4f}")


# Step 7: Function to Predict Top 5 Suitable Tree Types
def recommend_top_species(neighbourhood_name, top_n=5):
    # Encode the input neighbourhood name
    encoded_neighbourhood = le_neighbourhood.transform([neighbourhood_name])[0]
    
    # Use median values for other features for simplicity
    avg_location_type = X['LOCATION_TYPE'].mode()[0]
    avg_condition_percent = X['CONDITION_PERCENT'].median()
    avg_diameter = X['DIAMETER_BREAST_HEIGHT'].median()
    avg_tree_age = X['tree_age'].median()
    
    # Create the input data array
    input_data = pd.DataFrame([[encoded_neighbourhood, avg_location_type, avg_condition_percent, avg_diameter, avg_tree_age]], 
                              columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'CONDITION_PERCENT', 'DIAMETER_BREAST_HEIGHT', 'tree_age'])
    input_data = scaler.transform(input_data)
    
    # Get indices of the nearest neighbors
    distances, indices = knn.kneighbors(input_data, n_neighbors=15)
    
    # Retrieve the species of the nearest neighbors
    species_nearby = y_train.iloc[indices[0]]
    top_species = species_nearby.value_counts().head(top_n).index.tolist()
    
    return top_species

# Step 8: Test the Function
user_neighbourhood = "BELGRAVIA"  # Example input
try:
    top_species = recommend_top_species(user_neighbourhood, top_n=5)
    print(f"Top {len(top_species)} recommended species for {user_neighbourhood}: {top_species}")
except ValueError as e:
    print(f"Error: {e}. Please check if the neighbourhood name is valid.")