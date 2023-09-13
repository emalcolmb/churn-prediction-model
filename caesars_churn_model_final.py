import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib

def load_data(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print("Data read successfully.")
        return df
    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def preprocess_data(data):
    # Handle missing values, encoding
    data.fillna(0, inplace=True)  # Replace missing values with 0 (or choose an appropriate strategy)
    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    X = data.drop(['CustomerID', 'Churn'], axis=1)
    y = data['Churn']
    return X, y, label_encoders

def train_model(X_train, y_train, model_filename):
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model using joblib
    joblib.dump(model, model_filename)
    
    return model

def load_model(model_filename):
    # Load the model using joblib
    model = joblib.load(model_filename)
    return model

def predict_churn_probabilities(model, X):
    return model.predict_proba(X)[:, 1]

def main():
    excel_file_path = 'E Commerce Dataset.xlsx'
    sheet_name = 'E Comm'
    model_filename = 'xgboost_churn_model.joblib'  # Specify the model file name

    data = load_data(excel_file_path, sheet_name)
    if data is not None:
        X, y, label_encoders = preprocess_data(data)
        model = train_model(X, y, model_filename)  # Save the model using joblib

        # Predict churn probabilities for the entire dataset
        churn_probabilities = predict_churn_probabilities(model, X)

        # Add churn probabilities to the DataFrame
        data['Churn_Prediction_Probability'] = churn_probabilities

        # Sort the DataFrame by churn probability in descending order
        sorted_data = data.sort_values(by='Churn_Prediction_Probability', ascending=False)

        # Display the top N customers most likely to churn
        N = 1000  # Adjust this value to get the top N customers
        top_churners = sorted_data.head(N)

        print("Top {} Customers Most Likely to Churn:".format(N))
        print(top_churners[['CustomerID', 'Churn_Prediction_Probability']])

if __name__ == "__main__":
    main()
