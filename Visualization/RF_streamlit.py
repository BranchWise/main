import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import pandas as pd
import zipfile

# --- Helper Functions ---
def group_rare_categories(column, threshold):
    """Group rare categories in a categorical column based on the column's own counts."""
    counts = column.value_counts()
    return column.apply(lambda x: x if pd.notna(x) and counts[x] < threshold else 'Other')

def classify_condition(row):
    """Classify tree condition based on thresholds of age and condition percentage."""
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

def recommend_trees(neighborhood, data, model, top_n=5):
    """Recommend the names of top tree species (common names) for a given neighborhood based on model predictions."""
    if neighborhood not in data['NEIGHBOURHOOD_NAME'].unique():
        raise ValueError(f"Neighborhood '{neighborhood}' does not exist in the dataset.")
    
    # Filter data for the specified neighborhood and make a copy
    filtered_data = data[data['NEIGHBOURHOOD_NAME'] == neighborhood].copy()
    
    # Check for required columns before dropping
    required_columns = ['CONDITION_LABEL', 'SPECIES_COMMON']
    for col in required_columns:
        if col not in filtered_data.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
    
    X_filtered = filtered_data.drop(columns=required_columns)
    
    # Use the trained model to predict conditions for the filtered data
    predicted_conditions = model.predict(X_filtered)
    
    # Add predictions to the filtered data
    filtered_data['PREDICTED_CONDITION'] = predicted_conditions
    
    # Calculate scores for each species based on predicted GREAT or GOOD labels
    tree_scores = filtered_data.groupby('SPECIES_COMMON')['PREDICTED_CONDITION'].apply(
        lambda x: ((x == 'GREAT') | (x == 'GOOD')).sum()  # Count both GREAT and GOOD predictions
    ).sort_values(ascending=False)
    
    # Get the top species
    top_species = tree_scores.head(top_n).index.tolist()

    return top_species


# --- Load and preprocess data ---
@st.cache_data
def load_data():
    zip_path = 'Data.zip'

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('Edmonton_Trees_and_Neighbourhoods.csv') as f:
            data = pd.read_csv(f)

    data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
    current_year = pd.Timestamp.now().year
    data['TREE_AGE'] = current_year - data['PLANTED_DATE'].dt.year
    data = data.dropna(subset=['CONDITION_PERCENT', 'DIAMETER_BREAST_HEIGHT'])
    return data

def preprocess_data(data):
    # Train-Test Split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Fill missing TREE_AGE with median
    train_data['TREE_AGE'] = train_data['TREE_AGE'].fillna(train_data['TREE_AGE'].median())
    test_data['TREE_AGE'] = test_data['TREE_AGE'].fillna(train_data['TREE_AGE'].median())

    # Define condition classification
    train_data['CONDITION_LABEL'] = train_data.apply(classify_condition, axis=1)
    test_data['CONDITION_LABEL'] = test_data.apply(classify_condition, axis=1)
    return train_data, test_data

# --- Main App ---
def main():
    st.image("forest-header.png", use_container_width=True)
    st.title("Tree Recommendation System")

    # Load and preprocess data
    data = load_data()
    train_data, test_data = preprocess_data(data)

    with st.sidebar.expander("About this Program"):
        st.markdown(
        """
        This program was designed by the **Branchwise Team** as part of the **RBC Borealis AI Let's Solve It** program.  

        This initiative uses AI to assist **Edmonton Urban Planners** in identifying the most ideal tree species for specific neighborhoods using data from the **City of Edmonton**.
        """
        )

    st.sidebar.title("Input Options")
    
    neighborhood = st.sidebar.selectbox("Select a Neighborhood", train_data['NEIGHBOURHOOD_NAME'].unique())

    st.sidebar.write("Click below to get recommendations:")
    if st.sidebar.button("Get Recommendations"):
        try:
            st.markdown(
                f"<h2 style='font-size:28px; font-weight:bold; color:#2E7D32;'>Top recommended tree species for {neighborhood}:</h2>",
                unsafe_allow_html=True
            )
            # Train the model and generate recommendations
            best_model = train_random_forest(train_data)
            top_species = recommend_trees(neighborhood, train_data, best_model, top_n=5)
            st.markdown(
                "<ul style='font-size:20px; color:#2E7D32;'>"
                + "".join([f"<li>{tree}</li>" for tree in top_species])
                + "</ul>",
                unsafe_allow_html=True
            )
        except ValueError as e:
            st.error(e)
    # Add the logo to the sidebar
    st.sidebar.markdown("---")  # Optional divider line
    st.sidebar.image("branchwise_logo.png", use_container_width=True)  # Adjust the filename

def train_random_forest(train_data):
    useful_cols = ['NEIGHBOURHOOD_NAME', 'DIAMETER_BREAST_HEIGHT', 'TREE_AGE', 
                   'SPECIES_BOTANICAL', 'CONDITION_LABEL']
    X = train_data[useful_cols[:-1]]
    y = train_data['CONDITION_LABEL']
    
    # Encode categorical features
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
    
    # Fit the model
    pipeline.fit(X, y)
    return pipeline

if __name__ == "__main__":
    main()