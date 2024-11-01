import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
from shapely.ops import unary_union
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('/Users/sophiecabungcal/Downloads/Trees_20241025.csv')

# Preprocess data
data['CONDITION_PERCENT'] = data['CONDITION_PERCENT'].fillna(data['CONDITION_PERCENT'].mean())
data['DIAMETER_BREAST_HEIGHT'] = data['DIAMETER_BREAST_HEIGHT'].fillna(data['DIAMETER_BREAST_HEIGHT'].mean())
data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
data['PLANTED_DATE'] = data['PLANTED_DATE'].fillna(datetime.now())
data['tree_age'] = (datetime.now() - data['PLANTED_DATE']).dt.days // 365

# Convert LOCATION to geospatial data
data['geometry'] = data.apply(lambda row: Point(row['LONGITUDE'], row['LATITUDE']), axis=1)
gdf = gpd.GeoDataFrame(data, geometry='geometry')

# Ensure neighborhood names are lowercase
gdf['NEIGHBOURHOOD_NAME'] = gdf['NEIGHBOURHOOD_NAME'].str.lower()

# Aggregate points by neighborhood to create multipolygons
neighborhoods = gdf.dissolve(by='NEIGHBOURHOOD_NAME', as_index=False)
neighborhoods['geometry'] = neighborhoods['geometry'].apply(lambda x: unary_union(x) if isinstance(x, MultiPolygon) else x)

# Categorize tree conditions
def categorize_condition(condition):
    if condition >= 70:
        return "Great"
    elif 65 <= condition < 70:
        return "Good"
    elif condition == 65:
        return "Mediocre"
    else:
        return "Bad"

gdf['condition_category'] = gdf['CONDITION_PERCENT'].apply(categorize_condition)

# Filter for 'Great' and 'Good' trees
suitable_trees = gdf[gdf['condition_category'].isin(['Great', 'Good'])]

# Filter out classes with fewer than 2 members
class_counts = suitable_trees['SPECIES_COMMON'].value_counts()
suitable_trees = suitable_trees[suitable_trees['SPECIES_COMMON'].isin(class_counts[class_counts >= 2].index)]

# Feature selection
features = suitable_trees[['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'DIAMETER_BREAST_HEIGHT', 'tree_age', 'LATITUDE', 'LONGITUDE']]
target = suitable_trees['SPECIES_COMMON']

# One-hot encode categorical features
features = pd.get_dummies(features, columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE'], drop_first=True)

# Save the feature names used during training
feature_names = features.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Function to predict best tree species for a given coordinate or neighborhood
def predict_best_tree(neighbourhood_name=None, latitude=None, longitude=None):
    if neighbourhood_name:
        neighbourhood_name = neighbourhood_name.lower()
        input_data = pd.DataFrame({'NEIGHBOURHOOD_NAME': [neighbourhood_name], 'LOCATION_TYPE': ['Unknown'], 'DIAMETER_BREAST_HEIGHT': [0], 'tree_age': [0], 'LATITUDE': [0], 'LONGITUDE': [0]})
    elif latitude and longitude:
        input_data = pd.DataFrame({'NEIGHBOURHOOD_NAME': ['Unknown'], 'LOCATION_TYPE': ['Unknown'], 'DIAMETER_BREAST_HEIGHT': [0], 'tree_age': [0], 'LATITUDE': [latitude], 'LONGITUDE': [longitude]})
    else:
        return "Please provide either a neighbourhood name or coordinates."

    input_data = pd.get_dummies(input_data, columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE'], drop_first=True)
    
    # Align input data columns with training data columns
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    
    prediction = rf_model.predict(input_data)
    return prediction[0]

# Example usage
print("morin industrial:", predict_best_tree(neighbourhood_name='morin industrial'))
print("charlesworth:", predict_best_tree(neighbourhood_name='charlesworth'))
print("colverdale:", predict_best_tree(neighbourhood_name='colverdale'))