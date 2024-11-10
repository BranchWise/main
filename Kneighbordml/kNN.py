import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from datetime import datetime

# Load and preprocess the data
data = pd.read_csv('Trees_Modified_Species_Common.csv')
data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
data['tree_age'] = (datetime.now() - data['PLANTED_DATE']).dt.days // 365

# Step 1: Categorize tree condition and filter for "Great" and "Good" trees
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

print("Setting Age and Condition Stages...")

# Step 3: Encode categorical variables
le_location = LabelEncoder()
le_neighbourhood = LabelEncoder()

suitable_trees['LOCATION_TYPE'] = le_location.fit_transform(suitable_trees['LOCATION_TYPE'])
suitable_trees['NEIGHBOURHOOD_NAME'] = le_neighbourhood.fit_transform(suitable_trees['NEIGHBOURHOOD_NAME'])

X = suitable_trees[['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'CONDITION_PERCENT', 'DIAMETER_BREAST_HEIGHT', 'tree_age']]
y = suitable_trees['SPECIES_COMMON']

# Handle missing values
X = X.dropna()
y = y.loc[X.index]

print("Clustering...")

# Perform KMeans clustering on the feature data before the train-test split
num_clusters = 10  # Adjust this number based on experimentation
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
X['cluster'] = kmeans.fit_predict(X)

print("Data ready for Train-Testing...")

class_counts = y.value_counts()
valid_classes = class_counts[class_counts > 1].index  # Classes with more than 1 sample
X_filtered = X[y.isin(valid_classes)]
y_filtered = y[y.isin(valid_classes)]

# Now perform stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)

print("Feature Scaling...")

# Feature scaling
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Finding Optimal Best K...")

# Step 5: Train k-Nearest Neighbors Model
param_grid = {'n_neighbors': list(range(1, 8, 1))}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=2)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
print(f"Optimal n_neighbors found: {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Evaluate the ensemble model
y_pred = knn.predict(X_test)  
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Function to get top 5 recommended tree species
def recommend_top_species(neighbourhood_name, top_n=5):
    # Encode the input neighbourhood name
    encoded_neighbourhood = le_neighbourhood.transform([neighbourhood_name])[0]
    avg_location_type = X['LOCATION_TYPE'].mode()[0]
    avg_condition_percent = X['CONDITION_PERCENT'].median()
    avg_diameter = X['DIAMETER_BREAST_HEIGHT'].median()
    avg_tree_age = X['tree_age'].median()
    avg_cluster = X['cluster'].mode()[0]  # Use the most common cluster for simplicity

    # Create the input data array
    input_data = pd.DataFrame([[encoded_neighbourhood, avg_location_type, avg_condition_percent, avg_diameter, avg_tree_age, avg_cluster]], 
                              columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'CONDITION_PERCENT', 'DIAMETER_BREAST_HEIGHT', 'tree_age', 'cluster'])
    input_data = scaler.transform(input_data)
    
    # Get probability scores for each class (species) from the ensemble
    probabilities = knn.predict_proba(input_data)[0]
    
    # Map probabilities to class names
    class_probabilities = sorted(zip(knn.classes_, probabilities), key=lambda x: x[1], reverse=True)
    
    # Retrieve the top N species
    top_species = [species for species, prob in class_probabilities[:top_n]]
    
    return top_species

# Test the function
user_neighbourhood = "MAGRATH HEIGHTS"
try:
    top_species = recommend_top_species(user_neighbourhood, top_n=5)
    print(f"Top {len(top_species)} recommended species for {user_neighbourhood}: {top_species}")
except ValueError as e:
    print(f"Error: {e}. Please check if the neighbourhood name is valid.")