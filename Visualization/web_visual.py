import streamlit as st
from joblib import load
import pandas as pd
import zipfile

# Load and preprocess data
@st.cache_data
def load_data():

    zip_path = 'main/visualization/Data.zip'

    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('Edmonton_Trees_and_Neighbourhoods.csv') as f:
            data = pd.read_csv(f)

    data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
    current_year = pd.Timestamp.now().year
    data['TREE_AGE'] = current_year - data['PLANTED_DATE'].dt.year
    data = data.dropna(subset=['CONDITION_PERCENT', 'DIAMETER_BREAST_HEIGHT'])
    
    # Add the CONDITION_LABEL column
    data['CONDITION_LABEL'] = data.apply(classify_condition, axis=1)
    return data

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

# Recommend tree species
def recommend_trees(neighborhood, data, model, top_n=5):
    if neighborhood not in data['NEIGHBOURHOOD_NAME'].unique():
        raise ValueError(f"Neighborhood '{neighborhood}' does not exist in the dataset.")
    
    filtered_data = data[data['NEIGHBOURHOOD_NAME'] == neighborhood].copy()
    required_columns = ['CONDITION_LABEL', 'SPECIES_COMMON']
    for col in required_columns:
        if col not in filtered_data.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
    
    X_filtered = filtered_data.drop(columns=required_columns)
    predicted_conditions = model.predict(X_filtered)
    filtered_data['PREDICTED_CONDITION'] = predicted_conditions

    tree_scores = filtered_data.groupby('SPECIES_COMMON')['PREDICTED_CONDITION'].apply(
        lambda x: ((x == 'GREAT') | (x == 'GOOD')).sum()
    ).sort_values(ascending=False)
    
    top_species = tree_scores.head(top_n).index.tolist()
    return top_species

# Streamlit app
def main():
    st.image("./main/visualization/images/forest-header.png", use_container_width=True)
    st.title("Tree Recommendation System")

    # Load and preprocess data
    data = load_data()
    
    try:
        model = load('main/models/training/trained_tree_model.joblib')
        print("Model loaded successfully!")
    except FileNotFoundError:
        st.error("Model file not found! Train the model first.")
        return
    
    with st.sidebar.expander("About this Program"):
        st.markdown(
        """
        This program was designed by the **Branchwise Team** as part of the **RBC Borealis AI Let's Solve It** program.  

        This initiative uses AI to assist **Edmonton Urban Planners** in identifying the most ideal tree species for specific neighborhoods using data from the **City of Edmonton**.
        """
        )

    st.sidebar.title("Input Options")
    
    neighborhood = st.sidebar.selectbox("Select a Neighborhood", data['NEIGHBOURHOOD_NAME'].unique())

    st.sidebar.write("Click below to get recommendations:")
    if st.sidebar.button("Get Recommendations"):
        try:
            top_species = recommend_trees(neighborhood, data, model, top_n=5)
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
    st.sidebar.image("./main/visualization/images/branchwise_logo.png", use_container_width=True)  # Adjust the filename

if __name__ == "__main__":
    main()