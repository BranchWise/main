# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from local file
df = pd.read_csv('/Users/mylayambao/Downloads/edmonton_tree_map.csv') 

# Use random sampling to create a subset of 50,000 rows
subset_df = df.sample(n=50000, random_state=42)  # Use `n=50000` for an exact number of rows

# Select features and target from the subset
X = subset_df[['NEIGHBOURHOOD_NAME', 'SPECIES_COMMON', 'LOCATION_TYPE', 
               'DIAMETER_BREAST_HEIGHT', 'LATITUDE', 'LONGITUDE', 'OWNER', 
               'Bears Edible Fruit', 'Type of Edible Fruit']]
y = subset_df['CONDITION_PERCENT']

# One-hot encode the categorical features
X_encoded = pd.get_dummies(X, columns=['NEIGHBOURHOOD_NAME', 'SPECIES_COMMON', 
                                       'LOCATION_TYPE', 'OWNER', 
                                       'Bears Edible Fruit', 'Type of Edible Fruit'], 
                           drop_first=True)

# Save the trained columns for alignment during prediction
trained_columns = X_encoded.columns

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the MLPRegressor model with some example parameters
mlp = MLPRegressor(hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
                   activation='relu',             # Activation function
                   solver='adam',                 # Optimizer
                   max_iter=500,                  # Maximum number of epochs
                   early_stopping=True,           # Enable early stopping
                   validation_fraction=0.1,       # Use 10% of training data for validation
                   random_state=42)

# Train the MLPRegressor model
mlp.fit(X_train_scaled, y_train)

# Make predictions
y_pred = mlp.predict(X_test_scaled)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
accuracy = mlp.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)
