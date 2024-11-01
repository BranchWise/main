import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

file_path = '/Users/sophiecabungcal/Downloads/Trees_20241025.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
data['CONDITION_PERCENT'] = data['CONDITION_PERCENT'].fillna(data['CONDITION_PERCENT'].mean())
data['DIAMETER_BREAST_HEIGHT'] = data['DIAMETER_BREAST_HEIGHT'].fillna(data['DIAMETER_BREAST_HEIGHT'].mean())
data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
data['PLANTED_DATE'] = data['PLANTED_DATE'].fillna(datetime.now())
data['tree_age'] = (datetime.now() - data['PLANTED_DATE']).dt.days // 365  # Convert to years
data['NEIGHBOURHOOD_NAME'] = data['NEIGHBOURHOOD_NAME'].str.lower()  # Normalize to lowercase
data['LOCATION_TYPE'] = data['LOCATION_TYPE'].str.lower()  # Normalize to lowercase

def categorize_condition(condition):
    """
    Defines tree condition labels
    """
    if condition >= 70:
        return "Great"
    elif 65 <= condition < 70:
        return "Good"
    elif condition == 65:
        return "Mediocre"
    else:
        return "Bad"

data['condition_category'] = data['CONDITION_PERCENT'].apply(categorize_condition)

# Filter for 'Great' and 'Good' trees
suitable_trees = data[data['condition_category'].isin(['Great', 'Good'])]

# Filter out classes w/ <2 members
class_counts = suitable_trees['SPECIES_COMMON'].value_counts()
suitable_trees = suitable_trees[suitable_trees['SPECIES_COMMON'].isin(class_counts[class_counts >= 2].index)]

# Feature Selection
features = suitable_trees[['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'DIAMETER_BREAST_HEIGHT', 'tree_age', 'LATITUDE', 'LONGITUDE']]
target = suitable_trees['SPECIES_COMMON']

# One-hot encode categorical features
features = pd.get_dummies(features, columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE'], drop_first=True)

# Train-Test Split w/ Stratification
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Model Training & Eval
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importances = rf.feature_importances_
feature_names = features.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances:\n", importance_df)

def map_coordinates_to_neighbourhood(lat, lon, data):
    """
    Maps coordinates to a neighbourhood
    """
    neighbourhoods = {}
    for name, group in data.groupby('NEIGHBOURHOOD_NAME'):
        coords = list(zip(group['LONGITUDE'], group['LATITUDE']))
        if len(coords) >= 4:  # Ensure there are enough coordinates to form a valid polygon
            neighbourhoods[name] = Polygon(coords)
    
    point = Point(lon, lat)
    for neighbourhood, polygon in neighbourhoods.items():
        if polygon.contains(point):
            return neighbourhood
    return None

def format_species_name(species_name):
    """
    Converts species name from 'Ash, Green' to 'Green Ash'
    """
    parts = species_name.split(', ')
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return species_name

def recommend_top_species(neighbourhood_name=None, location_type=None, lat=None, lon=None, model=None, data=None, top_n=5):
    """
    Given a neighbourhood name, location type, and/or coordinates, return the top N recommended species.
    If location type is not provided, return the general top species for the neighbourhood.
    """
    if neighbourhood_name:
        neighbourhood_name = neighbourhood_name.lower()
    elif lat is not None and lon is not None:
        neighbourhood_name = map_coordinates_to_neighbourhood(lat, lon, data)
    
    if location_type:
        location_type = location_type.lower()
    
    if neighbourhood_name:
        if location_type:
            input_features = data[(data['NEIGHBOURHOOD_NAME'] == neighbourhood_name) & (data['LOCATION_TYPE'] == location_type)]
        else:
            input_features = data[data['NEIGHBOURHOOD_NAME'] == neighbourhood_name]
        
        if not input_features.empty:
            input_features = input_features.iloc[0:1]  # Take the first row for prediction
            input_features = pd.get_dummies(input_features, columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE'], drop_first=True)
            
            # Align input features with training features
            input_features = input_features.reindex(columns=model.feature_names_in_, fill_value=0)
            
            probabilities = model.predict_proba(input_features)
            species_prob = sorted(zip(model.classes_, probabilities[0]), key=lambda x: x[1], reverse=True)
            formatted_species = [format_species_name(species) for species, prob in species_prob[:top_n]]
            return formatted_species, neighbourhood_name, location_type
    return [], neighbourhood_name, location_type

def format_recommendation_output(species_list, neighbourhood_name=None, location_type=None, lat=None, lon=None, top_n=5):
    """
    Formats the output of the recommended species.
    """
    if neighbourhood_name:
        neighbourhood_name = neighbourhood_name.title()
    else:
        neighbourhood_name = f"the location at lat: {lat} lon: {lon}"
    
    if location_type:
        location_type = location_type.title()
        location_info = f"{neighbourhood_name} - {location_type}"
    else:
        location_info = neighbourhood_name
    
    output = f"The top {top_n} recommended trees for {location_info}:\n"
    for i, species in enumerate(species_list, 1):
        output += f"{i}. {species}\n"
    
    return output

# Ex:
# Predicting top 5 species for a given neighbourhood name
species_list, neighbourhood_name, location_type = recommend_top_species(neighbourhood_name="matt berry", model=rf, data=data, top_n=5)
print(format_recommendation_output(species_list, neighbourhood_name=neighbourhood_name, location_type=location_type, top_n=5))

# Predicting top 5 species for given coordinates
species_list, neighbourhood_name, location_type = recommend_top_species(lat=53.5461, lon=-113.4938, model=rf, data=data, top_n=5)
print(format_recommendation_output(species_list, lat=53.5461, lon=-113.4938, top_n=5))

# Predicting top 5 species for a given neighbourhood name and location type
species_list, neighbourhood_name, location_type = recommend_top_species(neighbourhood_name="charlesworth", location_type="park", model=rf, data=data, top_n=5)
print(format_recommendation_output(species_list, neighbourhood_name=neighbourhood_name, location_type=location_type, top_n=5))