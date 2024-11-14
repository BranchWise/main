import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

## Author: mylayambao ##

# Load the data from local file
data = pd.read_csv('/Users/mylayambao/Downloads/edmonton_tree_map.csv') 

# Select features and target
X = data[['NEIGHBOURHOOD_NAME','LOCATION_TYPE', 'DIAMETER_BREAST_HEIGHT', 'LATITUDE', 'LONGITUDE', 'OWNER', 'Bears Edible Fruit', 'Type of Edible Fruit']]
y = data['SPECIES_COMMON']  # Assuming we're predicting the common species

# One-hot encode the categorical features
X = pd.get_dummies(X, columns=['NEIGHBOURHOOD_NAME', 'LOCATION_TYPE', 'OWNER', 'Bears Edible Fruit', 'Type of Edible Fruit'], drop_first=True)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Gradient Boosting model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def recommend_top_species(input_features, model, top_n=5):
    """
    Given input features (location data), return the top N recommended species.
    """
    probabilities = model.predict_proba(input_features)
    species_prob = sorted(zip(model.classes_, probabilities[0]), key=lambda x: x[1], reverse=True)
    return [species for species, prob in species_prob[:top_n]]

# Example: Predicting top 5 species for the Belgravia neighbourhood
sample_input = X[X['NEIGHBOURHOOD_NAME_Belgravia'] == 1].head(1)
print("Top 5 recommended species for Belgravia neighbourhood:")
print(recommend_top_species(sample_input, model, top_n=5))