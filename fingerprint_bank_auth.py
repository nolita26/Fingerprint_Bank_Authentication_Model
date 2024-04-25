import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import os

app = Flask(__name__)
# run_with_ngrok(app)

# Load fingerprint image paths dataset from CSV
fp_data = pd.read_csv('/content/fingerprints.csv')
fp_data.head(10)

# Load bank account details dataset from CSV
bank_data = pd.read_csv('/content/bank.csv')
bank_data.head(10)

# Read the existing data from bank.csv into a DataFrame
bank_data = pd.read_csv("/content/bank.csv")

# Generate random account numbers for the 'Account No' column
# Assuming account numbers are integers and you want unique account numbers
# Adjust the range and length of account numbers as needed
account_numbers = [random.randint(1000000000, 9999999999) for _ in range(len(bank_data))]

# Assign the generated random account numbers to the 'Account No' column
bank_data['Account No'] = account_numbers

# Write the modified data back to bank.csv
bank_data.to_csv("bank.csv", index=False)

bank_data.head()

# Read the account numbers from bank.csv
bank_data = pd.read_csv("/content/bank.csv")

# Create a new column 'Account No' in fp_data and populate it with the account numbers
fp_data['Account No'] = bank_data['Account No']

# Write the modified data back to fingerprints.csv
fp_data.to_csv("fingerprints.csv", index=False)

fp_data.head()

fp_data.describe()
bank_data.describe()

print("Columns in fp_data:", fp_data.columns)
print("Columns in bank_data:", bank_data.columns)

# Assuming you have a common key (In this case, Account No) to merge the datasets
merged_data = pd.merge(fp_data, bank_data, on='Account No')

merged_data.head()

# Function to authenticate fingerprint
def authenticate_fingerprint(filename, account_no):
  if filename in merged_data['filename'].values:
        # Check if the account number matches
        if merged_data.loc[merged_data['filename'] == filename, 'Account No'].values[0] == account_no:
            return True  # Fingerprint authenticated
  return False

# Test the function
fingerprint_to_auth = '364__M_Right_little_finger.BMP'
account_no = 2344835938
is_auth = authenticate_fingerprint(fingerprint_to_auth, account_no)
print("Is fingerprint authenticated?", is_auth)

merged_data['authenticated'] = merged_data.apply(lambda row: authenticate_fingerprint(row['filename'], row['Account No']), axis=1)

# Feature Engineering
# Input features (fingerprint filename and account number)
X = merged_data[['filename', 'Account No']]

# Target labels (authentication status)
y = merged_data['authenticated']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input features to be two-dimensional
X_train_reshaped = X_train.apply(lambda x: len(x)).values.reshape(-1, 1)
X_test_reshaped = X_test.apply(lambda x: len(x)).values.reshape(-1, 1)

# Check for duplicates in X_train_reshaped and y_train
duplicates_X_train = len(np.unique(X_train_reshaped)) != len(X_train_reshaped)
duplicates_y_train = len(np.unique(y_train)) != len(y_train)

print("Duplicates in X_train_reshaped:", duplicates_X_train)
print("Duplicates in y_train:", duplicates_y_train)

# Remove duplicates from X_train_reshaped
X_train_reshaped = np.unique(X_train_reshaped, axis=0)

# Remove duplicates from y_train
y_train = np.unique(y_train)

# If duplicates exist, remove them
if duplicates_X_train or duplicates_y_train:
    X_train_reshaped_unique, indices = np.unique(X_train_reshaped, axis=0, return_index=True)
    y_train_unique = y_train[indices]

    print("Shape after removing duplicates from X_train_reshaped:", X_train_reshaped_unique.shape)
    print("Shape after removing duplicates from y_train:", y_train_unique.shape)
else:
    print("No duplicates found.")

# Train a Random Forest classifier (for demonstration)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train_reshaped, y_train)

# Define the API endpoint for fingerprint authentication
@app.route('/authenticate', methods=['POST'])
def authenticate_endpoints():
    data = request.json
    fingerprint_path = data.get('filename')
    account_number = data.get('Account No')
    if fingerprint_path and account_number:
        if os.path.exists(fingerprint_path):
            if authenticate_fingerprint(fingerprint_path, account_number):
                return jsonify({"authenticated": True})
    return jsonify({"authenticated": False})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')