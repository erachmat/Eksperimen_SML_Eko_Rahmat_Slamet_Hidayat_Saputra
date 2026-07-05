import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

def preprocess_data(input_path, output_path):
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df = df.drop(columns=['customerID'])

    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols),
            ('passthrough_bin', 'passthrough', ['SeniorCitizen'])
        ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_names) + ['SeniorCitizen']

    train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_df['Churn'] = y_train.values

    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df['Churn'] = y_test.values

    os.makedirs(output_path, exist_ok=True)
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)
    print(f"Successfully saved processed train and test sets to {output_path}")

if __name__ == "__main__":
    RAW_DATA_PATH = "telco_churn_raw/Telco-Customer-Churn.csv"
    PROCESSED_DIR = "preprocessing/telco_churn_preprocessing"
    preprocess_data(RAW_DATA_PATH, PROCESSED_DIR)
