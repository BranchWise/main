import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# load data
df = pd.read_csv('edmonton_trees.csv')

# select features and targets
X = df[['NEIGHBOURHOOD_NAME', 'SPECIES_COMMON', 'LOCATION_TYPE', 'DIAMETER_BREAST_HEIGHT', 'LATITUDE', 'LONGITUDE', 'OWNER', 'Bears Edible Fruit', 'Type of Edible Fruit']]
y = df['CONDITION_PERCENT']

# adding 'tree_age' based on the planting date
df['PLANTED_DATE'] = pd.to_datetime(df['PLANTED_DATE'], errors='coerce')
df['tree_age'] = (pd.Timestamp.now() - df['PLANTED_DATE']).dt.days // 365
X['tree_age'] = df['tree_age']

# outlier Detection and Handling
for feature in ['DIAMETER_BREAST_HEIGHT', 'tree_age']:
    Q1 = X[feature].quantile(0.25)
    Q3 = X[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # cap outliers to the lower and upper bounds
    X[feature] = np.where(X[feature] < lower_bound, lower_bound, X[feature])
    X[feature] = np.where(X[feature] > upper_bound, upper_bound, X[feature])

# one-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=['NEIGHBOURHOOD_NAME', 'SPECIES_COMMON', 'LOCATION_TYPE', 'OWNER', 'Bears Edible Fruit', 'Type of Edible Fruit'], drop_first=True)
trained_columns = X_encoded.columns

# split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# scaling Continuous Features with MinMaxScaler and RobustScaler
scalers = {'MinMaxScaler': MinMaxScaler(), 'RobustScaler': RobustScaler()}
for scaler_name, scaler in scalers.items():
    print(f"\nUsing {scaler_name}:")

    # fit and transform on train data, transform on test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # model Training with Elastic Net
    best_elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    best_elastic_net.fit(X_train_scaled, y_train)
    
    # predictions and Evaluation
    y_pred = best_elastic_net.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Mean Absolute Error (MAE): {mae}")

    try:
        mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100
        print(f"Accuracy (MAPE): {100 - mape:.2f}%")
    except ZeroDivisionError:
        print("Some values in y_test are zero; MAPE cannot be computed.")

# test the prediction function
def predict_condition_percent(neighbourhood_name, species_common, model, scaler, trained_columns, X):
    """
    Given input features species_common, neighbourhood_name predict the condition percentage.
    """
    # calculate averages/modes for other required features
    avg_location_type = X['LOCATION_TYPE'].mode()[0]
    avg_diameter = X['DIAMETER_BREAST_HEIGHT'].median()
    avg_latitude = X['LATITUDE'].median()
    avg_longitude = X['LONGITUDE'].median()
    avg_owner = X['OWNER'].mode()[0]
    avg_bears_edible_fruit = X['Bears Edible Fruit'].mode()[0]
    avg_type_of_edible_fruit = X['Type of Edible Fruit'].mode()[0]
    avg_tree_age = X['tree_age'].median()

    # construct input row with provided and averaged values
    input_data = {
        'NEIGHBOURHOOD_NAME': neighbourhood_name,
        'LOCATION_TYPE': avg_location_type,
        'DIAMETER_BREAST_HEIGHT': avg_diameter,
        'LATITUDE': avg_latitude,
        'LONGITUDE': avg_longitude,
        'OWNER': avg_owner,
        'Bears Edible Fruit': avg_bears_edible_fruit,
        'Type of Edible Fruit': avg_type_of_edible_fruit,
        'SPECIES_COMMON': species_common,
        'tree_age': avg_tree_age
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

# sample test for the function
sample_neighbourhood = 'EDGEMONT'
sample_species = 'Maple, Amur'
predicted_condition = predict_condition_percent(sample_neighbourhood, sample_species, best_elastic_net, scaler, trained_columns, X)
print(f"Predicted condition percentage for {sample_species} in {sample_neighbourhood}: {predicted_condition}")
