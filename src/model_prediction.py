import pandas as pd
import argparse
import pickle
import json
import numpy as np

def load_data(file_path):
    # TODO: Load test data from CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded test data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def load_model(model_path):
    # TODO: Load the trained model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def make_predictions(df, model):
    # Make sure that 'Total_minutes' is in the test
    if "Hour" in df.columns and "Minutes" in df.columns:
        df["Total_minutes"] = df["Hour"] * 60 + df["Minutes"]

    # Check for missing necessary columns
    missing_features = set(model.feature_names_in_) - set(df.columns)
    if missing_features:
        raise ValueError(f"Columns missing in test.csv: {missing_features}")

    # Make sure we use the same columns as in training
    X_test = df[model.feature_names_in_]

    # Make preds
    predictions = model.predict(X_test)
    return predictions

def save_predictions(predictions, predictions_file):
    # Convert predictions (numpy array) to the required dictionary format
    predictions_dict = {
        "target": {i+1: int(pred) for i, pred in enumerate(predictions)}
    }
    
    # Save as JSON
    with open(predictions_file, "w") as f:
        json.dump(predictions_dict, f, indent=4)

    print(f"Predictions saved successfully to {predictions_file}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test_data.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    if df is not None:
        model = load_model(model_file)
        if model is not None:
            predictions = make_predictions(df, model)
            save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
