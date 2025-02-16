import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # TODO: Load data from CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    # TODO: Handle missing values, outliers, etc.
    # Drop duplicate rows if any
    df = df.drop_duplicates()

    # Fill missing values (if applicable)
    df = df.fillna(df.median()) # Replace NaNs with column medians

    print("Data cleaned successfully.")
    return df

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.
    # Normalize sensor data
    sensor_cols = ['Sensor_alpha', 'Sensor_beta', 'Sensor_gamma']
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    # Convert Hour & Minutes into a single feater (e.g., total minutes per day)
    df['Total_minutes'] = df['Hour'] * 60 + df['Minutes']
    df.drop(columns=['Hour', 'Minutes'], inplace = True) # Drop original time columns

    print("Data processing completed.")
    return df

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    try:
        df.to_csv(output_file, index = False)
        print(f"Processed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)