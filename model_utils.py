from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

def build_mlp_model(hp, input_shape, num_classes):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(shape=(input_shape,)))
    
    # Tune the number of layers
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=1024, step=32),
            activation='relu'
        ))
        model.add(keras.layers.Dropout(hp.Float(f'dropout_{i}', 0, 0.5, step=0.1)))
    
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_data(years, base_dir = "."):
    police_csv_files = [f"{base_dir}/data/police/policecalls{year}.csv.neighborhood.csv" for year in years]

    # List to hold dataframes
    police_dfs = []

    # Loop through the list of files and read them into dataframes
    for file in police_csv_files:
        df = pd.read_csv(file)
        police_dfs.append(df)

    # Concatenate all dataframes into one
    police_df = pd.concat(police_dfs, ignore_index=True)
    return police_df

def transform_offense_date(df):
    # Convert 'OFFENSE_DATE' to datetime to extract time-related features
    df['OFFENSE_DATE'] = pd.to_datetime(df['OFFENSE_DATE'], format='%m/%d/%Y %I:%M:%S %p')

    # and adding time-related features
    df['OFFENSE_HOUR'] = df['OFFENSE_DATE'].dt.hour
    df['OFFENSE_DAY_OF_WEEK'] = df['OFFENSE_DATE'].dt.dayofweek
    df['OFFENSE_MONTH'] = df['OFFENSE_DATE'].dt.month
    df['OFFENSE_YEAR'] = df['OFFENSE_DATE'].dt.year

    # Define the date COVID started
    covid_start_date = pd.to_datetime('2020-03-16')

    # Create a new column 'AFTER_COVID' that is true if the OFFENSE_DATE is on or after the covid_start_date, otherwise false
    df['AFTER_COVID'] = df['OFFENSE_DATE'] >= covid_start_date
    df['AFTER_COVID'] = df['AFTER_COVID'].apply(lambda x: 'Yes' if x else 'No')

def calc_dispo_subset(df):
    # List of target disposition codes to keep as individual classes
    target_dispo_codes = ['A', 'B', 'C']

    dispo_mapping = {
        'A': 'Arrest Made',
        'B': 'Arrest by Warrant',
        'C': 'Criminal Citation',
        'Other': 'Other'
    }

    # Create a new target variable where specified disposition codes remain,
    # and all others are consolidated into a single class.
    df['DISPO_SUBSET'] = np.where(df['FINAL_DISPO_CODE'].isin(target_dispo_codes), 
                                            df['FINAL_DISPO_CODE'], 'Other')
    
    return dispo_mapping

def clean_and_transform_data(df):
    """
    This function performs data cleaning and transformation on the Police Calls dataset.

    Steps:
    1. Drop unnecessary columns:
        - Columns like 'CDTS', 'EID', 'CALL_NUMBER', 'START_DATE', 'REPORT_DATE', 'CITY', 'STATE', 
          'CALL_TYPE', 'FINAL_DISPO', 'FINAL_DISPO_CODE', and 'ADDRESS' are removed as they are not 
          needed for the analysis or model building.
    
    2. Remove rows with any missing values:
        - Drop all rows that contain missing values to ensure the dataset is clean and ready for analysis.
        - Print the number of rows before and after cleaning to keep track of how many records were removed.
    
    3. Drop the 'OFFENSE_DATE' and 'OFFENSE_TIME' columns:
        - These columns are dropped after any necessary features have been extracted, as they are no longer needed.

    4. Calculate the Euclidean distance from the center of San Jose:
        - Using San Jose's approximate geographical center (Latitude: 37.3382, Longitude: -121.8863),
          calculate the Euclidean distance from the center for each record based on its 'LATITUDE' and 'LONGITUDE'.
    
    5. Normalize the latitude and longitude:
        - Latitude and longitude values are normalized using StandardScaler to prepare for feature engineering.
    
    6. Generate polynomial features:
        - Using PolynomialFeatures with degree=2, generate additional polynomial terms (e.g., interaction terms)
          from the normalized latitude and longitude. This helps capture non-linear relationships in the data.
        - Only the interaction terms are used (skipping the first two columns), and the new features are
          concatenated back into the original DataFrame.

    7. Return the cleaned and transformed dataset, which includes normalized geographical features, 
       additional polynomial features, and the distance from the city center.
    """

    # Dropping unnecessary columns 
    df = df.drop( 
        columns=['CDTS', 'EID', 'CALL_NUMBER', 'START_DATE', 'REPORT_DATE', 'CITY', 'STATE', 
                'CALL_TYPE', 'FINAL_DISPO', 'FINAL_DISPO_CODE', 'ADDRESS'])
    
    # Remove rows with any missing values from the Police Calls dataset
    clean_df = df.dropna()
    print(f"Police Calls Dataset: {df.shape[0]} rows before cleaning, {clean_df.shape[0]} rows after cleaning.")

    # Drop 'OFFENSE_DATE' now that features have been extracted
    clean_df = clean_df.drop(columns=['OFFENSE_DATE', 'OFFENSE_TIME'])

    # Distance from center: San Jose's approximate center (Latitude and Longitude)
    SJ_CENTER_LAT, SJ_CENTER_LON = 37.3382, -121.8863

    # Calculate Euclidean distance from the city center
    clean_df['DISTANCE_FROM_CENTER'] = np.sqrt(
        (clean_df['LATITUDE'] - SJ_CENTER_LAT) ** 2 + 
        (clean_df['LONGITUDE'] - SJ_CENTER_LON) ** 2)

    # Normalize latitude and longitude before adding polynomial features
    scaler = StandardScaler()
    clean_df[['LATITUDE', 'LONGITUDE']] = scaler.fit_transform(clean_df[['LATITUDE', 'LONGITUDE']])

    # Generate polynomial features from normalized latitude and longitude
    poly = PolynomialFeatures(degree=2, include_bias=False)
    lat_lon_poly = poly.fit_transform(clean_df[['LATITUDE', 'LONGITUDE']])

    # Create a DataFrame with the polynomial features and concatenate with original data
    lat_lon_poly_df = pd.DataFrame(lat_lon_poly[:, 2:], 
                                columns=poly.get_feature_names_out(['LATITUDE', 'LONGITUDE'])[2:], 
                                index=clean_df.index)

    clean_df = pd.concat([clean_df, lat_lon_poly_df], axis=1)

    return clean_df

def split_data(df, mapping_dict, encoder=None):
    # Define features and target
    X = df.drop(columns=['DISPO_SUBSET'])
    y = df['DISPO_SUBSET']

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f'classes = {label_encoder.classes_}')

    label_names = [mapping_dict[code] for code in label_encoder.classes_]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print(X_train.shape)
    print(X_test.shape)

    class_labels = [mapping_dict[code] for code in label_encoder.inverse_transform([0, 1, 2, 3])]
    print(class_labels)

    # Identify numeric and categorical columns
    numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = ['CALLTYPE_CODE', 'neighborhood', 'AFTER_COVID']

    # Process categorical features
    if encoder is None:
        encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        encoder.fit(X_train[categorical_columns])

    # Transform categorical features
    X_train_cat = encoder.transform(X_train[categorical_columns])
    X_test_cat = encoder.transform(X_test[categorical_columns])

    # Get feature names for one-hot encoded columns
    onehot_feature_names = encoder.get_feature_names_out(categorical_columns)

    # Process numeric features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numeric_columns])
    X_test_num = scaler.transform(X_test[numeric_columns])

    # Combine numeric and categorical features
    X_train_scaled = np.hstack([X_train_num, X_train_cat])
    X_test_scaled = np.hstack([X_test_num, X_test_cat])

    # Combine feature names
    all_feature_names = numeric_columns + list(onehot_feature_names)

    # Convert the transformed data back to DataFrame with proper column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=all_feature_names, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=all_feature_names, index=X_test.index)

    # Get the class distribution for y_train
    sample_sizes = Counter(y_train)
    print(sample_sizes)

    # Adjust the class distribution (example: make class 3 equal to the sum of the other classes)
    sample_sizes[3] = sum(sorted(list(sample_sizes.values()))[0:3])
    print(sample_sizes)

    # Apply undersampling to balance the classes
    rus = RandomUnderSampler(sampling_strategy=sample_sizes, random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled_df, y_train)

    # Return the processed data and the encoder
    return (X_train_scaled_df, X_test_scaled_df, y_train, y_test,
            label_names, class_labels, X_train_resampled, y_train_resampled, encoder)



if __name__ == "__main__":
    print("Hello, world")