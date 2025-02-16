import pandas as pd
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def load_data(file_path):
    # TODO: Load processed data from CSV file
    try:
        df = pd.read_csv(file_path)
        print(F"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def split_data(df):
    # TODO: Split data into training and validation sets (the test set is already provided in data/test_data.csv)
    # Assuming 'Insect' is the target variable
    X = df.drop(columns=['Insect'])
    y = df['Insect']

    # Splitting into training (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Data successfully split into training and validation sets.")
    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
    # TODO: Initialize your model and train it (Random Forest Classifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Model training complete.")
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate model performance using F1 Score."""
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='weighted') # Weighted F1 score
    print(f"Validation F1 Score: {f1:.4f}")
    return f1

def save_model(model, model_path):
    # TODO: Save your trained model
    # Save as a picke file
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file):
    df = load_data(input_file)
    if df is not None:
        X_train, X_val, y_train, y_val = split_data(df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_val, y_val)  # Evaluate before saving
        save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)