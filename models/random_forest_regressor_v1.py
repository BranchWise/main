import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

"""
Author: mylayambao
"""

# Load the data from local file
df = pd.read_csv('/Users/mylayambao/Downloads/edmonton_tree_map.csv') 

# Select features and target
X = df[['NEIGHBOURHOOD_NAME','SPECIES_COMMON','LOCATION_TYPE', 'DIAMETER_BREAST_HEIGHT', 'LATITUDE', 'LONGITUDE', 'OWNER', 'Bears Edible Fruit', 'Type of Edible Fruit']]
y = df['CONDITION_PERCENT']

# One-hot encode the categorical features
X_encoded = pd.get_dummies(X, columns=['NEIGHBOURHOOD_NAME','SPECIES_COMMON' ,'LOCATION_TYPE', 'OWNER', 'Bears Edible Fruit', 'Type of Edible Fruit'], drop_first=True)

# Save the trained columns for alignment during prediction
trained_columns = X_encoded.columns

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Fit a scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Refit the model on the scaled data
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
print("Making predictions....")

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
accuracy = model.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)

def predict_condition_percent(neighbourhood_name, species_common, model, scaler, trained_columns,X):
    """
    Given input features species_common, neighbourhood_name predict the condition percentage.
    """
    # Calculate averages/modes for other required features
    avg_location_type = X['LOCATION_TYPE'].mode()[0]
    avg_diameter = X['DIAMETER_BREAST_HEIGHT'].median()
    avg_latitude = X['LATITUDE'].median()
    avg_longitude = X['LONGITUDE'].median()
    avg_owner = X['OWNER'].mode()[0]
    avg_bears_edible_fruit = X['Bears Edible Fruit'].mode()[0]
    avg_type_of_edible_fruit = X['Type of Edible Fruit'].mode()[0]

    # Construct input row with provided and averaged values
    input_data = {
        'NEIGHBOURHOOD_NAME': neighbourhood_name,
        'LOCATION_TYPE': avg_location_type,
        'DIAMETER_BREAST_HEIGHT': avg_diameter,
        'LATITUDE': avg_latitude,
        'LONGITUDE': avg_longitude,
        'OWNER': avg_owner,
        'Bears Edible Fruit': avg_bears_edible_fruit,
        'Type of Edible Fruit': avg_type_of_edible_fruit,
        'SPECIES_COMMON': species_common
    }


    # create a dataframe with the input features
    input_df = pd.DataFrame([input_data])
    # one-hot encode the categorical features
    input_df_encoded = pd.get_dummies(input_df, columns=['NEIGHBOURHOOD_NAME', 'SPECIES_COMMON', 'LOCATION_TYPE', 'OWNER', 'Bears Edible Fruit', 'Type of Edible Fruit'], drop_first=True)
    # Align the input dataframe with the trained columns
    input_df_encoded = input_df_encoded.reindex(columns=trained_columns, fill_value=0)
    # Scale the input features
    input_scaled = scaler.transform(input_df_encoded)
    # get the predicted condition percentage
    condition_percent = model.predict(input_scaled)[0]
    return condition_percent

# Test the function
sample_neighbourhood = 'EDGEMONT'
sample_species = 'Maple, Amur'
predicted_condition = predict_condition_percent(sample_neighbourhood, sample_species, model, scaler,trained_columns, X)
print(f"Predicted condition percentage for {sample_species} in {sample_neighbourhood}: {predicted_condition}")



