import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

## Author: mylayambao ##

# record the time it takes to run the model
import time
start_time = time.time()

# Load the data from local file 
data = pd.read_csv('C:\\Users\\cengl\\Downloads\\Trees_20241125.csv') 
print("RUNNING GBC MODEL")

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
# print training error
print("Training error:", 1 - model.score(X_train, y_train))
# print test error
print("Test error:", 1 - model.score(X_test, y_test))
# print precision, recall, f1-score se `zero_division` parameter to control this behavior.
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1 score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))
#print aoc and roc 

# Compute ROC curve and ROC area for each class
y_prob = model.predict_proba(X_test)
y_test_dummies = pd.get_dummies(y_test)

# Ensure y_prob has the same number of columns as y_test_dummies
if y_prob.shape[1] != y_test_dummies.shape[1]:
    raise ValueError("Mismatch between number of classes in y_test and y_prob")

roc_auc = roc_auc_score(y_test_dummies, y_prob, multi_class='ovr')
print("ROC AUC Score:", roc_auc)

# Plot ROC curve for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(model.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_dummies.iloc[:, i], y_prob[:, i])
    roc_auc[i] = roc_auc_score(y_test_dummies.iloc[:, i], y_prob[:, i])

plt.figure()
for i in range(len(model.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {model.classes_[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# print the time it ttakes to run the model
print("--- %s seconds ---" % (time.time() - start_time))


def recommend_top_species(input_features, model, top_n=5):
    """
    Given input features (location data), return the top N recommended species.
    """
    probabilities = model.predict_proba(input_features)
    species_prob = sorted(zip(model.classes_, probabilities[0]),   key=lambda x: x[1], reverse=True)
    return [species for species, prob in species_prob[:top_n]]

# Example: Predicting top 5 species for the Belgravia neighbourhood
sample_input = X[X['NEIGHBOURHOOD_NAME_Belgravia'] == 1].head(1)
print("Top 5 recommended species for Belgravia neighbourhood:")
print(recommend_top_species(sample_input, model, top_n=5))