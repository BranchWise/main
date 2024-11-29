import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

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

# --- Data Loading ---
print("Load data...")
data = pd.read_csv('Trees_Modified_Species_Common.csv')  # Confirm the file path

# --- Data Preprocessing ---
# Calculate tree age
data['PLANTED_DATE'] = pd.to_datetime(data['PLANTED_DATE'], errors='coerce')
current_year = pd.Timestamp.now().year
data['TREE_AGE'] = current_year - data['PLANTED_DATE'].dt.year

# Handle missing values
data = data.dropna(subset=['CONDITION_PERCENT', 'DIAMETER_BREAST_HEIGHT'])  # Drop critical missing rows

# --- Train-Test Split ---
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# ---------- needs to be fixed to only use the median in train set not full data
# Fill missing TREE_AGE with median from train data
train_data['TREE_AGE'] = train_data['TREE_AGE'].fillna(train_data['TREE_AGE'].median())
test_data['TREE_AGE'] = test_data['TREE_AGE'].fillna(train_data['TREE_AGE'].median())


# Define condition classification
train_data['CONDITION_LABEL'] = train_data.apply(classify_condition, axis=1)
test_data['CONDITION_LABEL'] = test_data.apply(classify_condition, axis=1)

X_train = train_data.drop('CONDITION_LABEL', axis=1)
y_train = train_data['CONDITION_LABEL']
X_test = test_data.drop('CONDITION_LABEL', axis=1)
y_test = test_data['CONDITION_LABEL']


# ---------- needs to be fixed to only reduce cardinality on categories in train set not full data
# Reduce high-cardinality categories
X_train['NEIGHBOURHOOD_NAME'] = group_rare_categories(X_train['NEIGHBOURHOOD_NAME'], 5)  # Group rare neighborhoods (less than 5)
X_train['SPECIES_BOTANICAL'] = group_rare_categories(X_train['SPECIES_BOTANICAL'], 10)    # Group rare neighborhoods (less than 10)

train_neighbourhoods = X_train['NEIGHBOURHOOD_NAME']
train_species = X_train['SPECIES_BOTANICAL']


X_test['NEIGHBOURHOOD_NAME'] = X_test['NEIGHBOURHOOD_NAME'].apply(lambda x: x if x in train_neighbourhoods.values else 'Other')
X_test['SPECIES_BOTANICAL'] = X_test['SPECIES_BOTANICAL'].apply(lambda x: x if x in train_species.values else 'Other')

#------------ how did they get useful columns?
# Filter for useful columns (include SPECIES_COMMON)
useful_cols = ['NEIGHBOURHOOD_NAME', 'DIAMETER_BREAST_HEIGHT', 'TREE_AGE', 
               'SPECIES_BOTANICAL', 'SPECIES_COMMON']

#feature selection
X_train = X_train[useful_cols]
X_test = X_test[useful_cols]


# Encode categorical features
categorical_features = ['NEIGHBOURHOOD_NAME', 'SPECIES_BOTANICAL']
numerical_features = ['DIAMETER_BREAST_HEIGHT', 'TREE_AGE']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# --- Feature Selection ---
# Optional: Filter low-variance features (if any)
def low_variance_filter(X, threshold=0.01):
    sel = VarianceThreshold(threshold=threshold)
    return sel.fit_transform(X)

# --- Feature Selection ---
# Optional: Filter low-variance features (if any)
def low_variance_filter(X, threshold=0.01):
    sel = VarianceThreshold(threshold=threshold)
    return sel.fit_transform(X)

# --- Model Training ---
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning with reduced iterations and folds
param_grid = {
    'classifier__n_estimators': [10, 50],  # Fewer estimators for faster tuning
    'classifier__max_depth': [None, 10],
    'classifier__min_samples_split': [2, 5]
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=8,  # Limited iterations
    cv=2,  # Reduced folds for faster tuning
    scoring='accuracy',
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

# Fit with subsampling to reduce dataset size during tuning
sample_data = train_data.sample(frac=0.2, random_state=42)  # Use 20% of the dataset
X_sample = sample_data.drop('CONDITION_LABEL', axis=1)
y_sample = sample_data['CONDITION_LABEL']
random_search.fit(X_sample, y_sample)

# --- Model Evaluation ---
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model on the training set
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_error = 1 - train_accuracy
print("Training Accuracy:", train_accuracy)
print("Training Error:", train_error)

# Evaluate the model on the testing set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_error = 1 - test_accuracy
print("Testing Accuracy:", test_accuracy)
print("Testing Error:", test_error)

# Evaluate the model on the full dataset
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# --- Learning Curve ---
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
plt.plot(train_sizes, test_scores_mean, label='Testing Accuracy')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()


# --- Recommendation System ---
def recommend_trees(neighborhood, data, model, top_n=5):
    """Recommend the names of top tree species (common names) for a given neighborhood based on model predictions."""
    if neighborhood not in data['NEIGHBOURHOOD_NAME'].unique():
        raise ValueError(f"Neighborhood '{neighborhood}' does not exist in the dataset.")
    
    # Filter data for the specified neighborhood and make a copy
    filtered_data = data[data['NEIGHBOURHOOD_NAME'] == neighborhood].copy()
    
    # Prepare input features for the model
    X_filtered = filtered_data.drop(columns=['CONDITION_LABEL', 'SPECIES_COMMON'])
    
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


# EXAMPLE CASE
neighborhood = "The Orchards at Ellerslie".upper()  # Convert input to uppercase for uniformity
try:
    print(f"Top {5} tree species for {neighborhood}:")
    top_species = recommend_trees(neighborhood, data, best_model)
    print("\n".join(top_species))  # Print each species on a new line
except ValueError as e:
    print(e)