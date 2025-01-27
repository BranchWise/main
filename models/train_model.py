import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
import zipfile

# Helper function to classify tree condition
def classify_condition(row):
    age = row['TREE_AGE']
    condition = row['CONDITION_PERCENT']
    if condition == 0 and age < 50:
        return 'VERY POOR'
    elif condition <= 50 and age < 50:
        return 'POOR'
    elif condition <= 30 and age >= 50:
        return 'POOR'
    elif 51 <= condition <= 65 and age < 50:
        return 'MEDIOCRE'
    elif 31 <= condition <= 50 and age >= 50:
        return 'MEDIOCRE'
    elif 66 <= condition <= 79 and age < 50:
        return 'GOOD'
    elif 51 <= condition <= 70 and age >= 50:
        return 'GOOD'
    elif condition >= 80 and age < 50:
        return 'GREAT'
    elif condition >= 71 and age >= 50:
        return 'GREAT'
    return 'UNKNOWN'

# Load and preprocess data
def load_data():

    zip_path = 'main/Visualization/Data.zip'

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('Edmonton_Trees_and_Neighbourhoods.csv') as f:
            data = pd.read_csv(f)

    data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
    current_year = pd.Timestamp.now().year
    data['TREE_AGE'] = current_year - data['PLANTED_DATE'].dt.year
    data = data.dropna(subset=['CONDITION_PERCENT', 'DIAMETER_BREAST_HEIGHT'])
    return data

def preprocess_data(data):
    data['TREE_AGE'] = data['TREE_AGE'].fillna(data['TREE_AGE'].median())
    data['CONDITION_LABEL'] = data.apply(classify_condition, axis=1)
    return data

def train_and_save_model(data):
    useful_cols = ['NEIGHBOURHOOD_NAME', 'DIAMETER_BREAST_HEIGHT', 'TREE_AGE', 'SPECIES_BOTANICAL', 'CONDITION_LABEL']
    X = data[useful_cols[:-1]]
    y = data['CONDITION_LABEL']

    # Preprocessing
    categorical_features = ['NEIGHBOURHOOD_NAME', 'SPECIES_BOTANICAL']
    numerical_features = ['DIAMETER_BREAST_HEIGHT', 'TREE_AGE']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train the model
    pipeline.fit(X, y)

    # Save the trained model
    dump(pipeline, 'main/models/training/trained_tree_model.joblib')
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    data = load_data()
    preprocessed_data = preprocess_data(data)
    train_and_save_model(preprocessed_data)