# Fingerprint_Bank_Authentication_Model

Built a fingerprint-based authentication system for users to access their bank accounts by skipping traditional methods! This project combines machine learning (Random Forest Algorithm) with secure biometric authentication to ensure seamless and secure user authentication.

How it Works:
- Data Preparation: I collected fingerprint image paths and bank account details from CSV files.
- Data Processing: After merging the datasets based on a common key (in this case, Account No), I used a Random Forest classifier to train the model.
- Model Training: The Random Forest model was trained using fingerprint image paths and account numbers as input features to predict authentication status.
- API Integration: I deployed a Flask-based API endpoint that receives fingerprint paths and account numbers, verifies the authenticity, and returns the authentication status.
