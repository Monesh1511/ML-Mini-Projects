# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
dd = pd.read_csv("diabetes.csv")

# Separate features (X) and target variable (y)
X = dd.drop(columns='Outcome', axis=1)
y = dd['Outcome']

# Standardize the data
scalar = StandardScaler()  # Create a StandardScaler object
scalar.fit(X)              # Fit the scaler on feature data
standard_dd = scalar.transform(X)  # Transform feature data
print(standard_dd)         # Print standardized values
X = standard_dd            # Replace original X with standardized version

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=40
)

# Create and train the SVM classifier (Support Vector Machine)
classifier = svm.SVC(kernel='linear')  # Linear kernel
classifier.fit(X_train, y_train)

# Predict on training data and calculate training accuracy
X_train_prediction = classifier.predict(X_train)
train_training_accuracy = accuracy_score(X_train_prediction, y_train)
print("ACCURACY", train_training_accuracy)

# Predict on testing data and calculate testing accuracy
X_test_prediction = classifier.predict(X_test)
test_training_accuracy = accuracy_score(X_test_prediction, y_test)
print("ACCURACY", test_training_accuracy)

# Use sample input data for prediction
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)  # Example input
setdata_array_usingnumpy = np.asarray(input_data)  # Convert to NumPy array
reshape_data = setdata_array_usingnumpy.reshape(1, -1)  # Reshape for single sample
std_data = scalar.transform(reshape_data)  # Standardize input
predicton = classifier.predict(std_data)  # Predict using trained model
print(predicton)  # Output: 1 = diabetic, 0 = not diabetic

# Get input from user for custom prediction
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
user_input = []

# Ask user to enter values for all features
for feature in feature_names:
    value = float(input(f"Enter {feature}: "))
    user_input.append(value)

# Create a DataFrame from user input
test_df = pd.DataFrame([user_input], columns=dd.drop(columns='Outcome').columns)

# Standardize the input
test_std = scalar.transform(test_df)

# Make prediction and print result
test_pred = classifier.predict(test_std)

if test_pred[0] == 1:
    print("The patient is diabetic.")
else:
    print("The patient is not diabetic.")
