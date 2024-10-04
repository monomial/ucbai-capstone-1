from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
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

def load_data(years):
    police_csv_files = [f"data/police/policecalls{year}.csv.neighborhood.csv" for year in years]

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

    # Create a new column 'AFTER_COVID' that is 1 if the OFFENSE_DATE is on or after the covid_start_date, otherwise 0
    df['AFTER_COVID'] = (df['OFFENSE_DATE'] >= covid_start_date).astype(int)

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


def clean_and_transform_data(df, encoder=None):
    # Remove rows with any missing values from the Police Calls dataset
    clean_df = df.dropna()
    print(f"Police Calls Dataset: {df.shape[0]} rows before cleaning, {clean_df.shape[0]} rows after cleaning.")

    # Dropping unnecessary columns 
    clean_df = clean_df.drop( \
        columns=['CDTS', 'EID', 'CALL_NUMBER', 'START_DATE', 'REPORT_DATE', 'CITY', 'STATE', \
                'CALL_TYPE', 'FINAL_DISPO', 'FINAL_DISPO_CODE', 'ADDRESS'])

    # Drop 'OFFENSE_DATE' now that features have been extracted
    clean_df = clean_df.drop(columns=['OFFENSE_DATE', 'OFFENSE_TIME'])

    # Distance from center: San Jose's approximate center (Latitude and Longitude)
    SJ_CENTER_LAT, SJ_CENTER_LON = 37.3382, -121.8863

    # Calculate Euclidean distance from the city center
    clean_df['DISTANCE_FROM_CENTER'] = np.sqrt(
        (clean_df['LATITUDE'] - SJ_CENTER_LAT) ** 2 + 
        (clean_df['LONGITUDE'] - SJ_CENTER_LON) ** 2)

    # Normalize latitude and longitude before adding polynomial featuers
    scaler = StandardScaler()
    clean_df[['LATITUDE', 'LONGITUDE']] = scaler.fit_transform(clean_df[['LATITUDE', 'LONGITUDE']])

    # Generate polynomial features from normalized latitude and longitude
    poly = PolynomialFeatures(degree=2, include_bias=False)
    lat_lon_poly = poly.fit_transform(clean_df[['LATITUDE', 'LONGITUDE']])

    # Create a DataFrame with the polynomial features and concatenate with original data
    lat_lon_poly_df = pd.DataFrame(lat_lon_poly[:, 2:], \
                                columns=poly.get_feature_names_out(['LATITUDE', 'LONGITUDE'])[2:], \
                                index=clean_df.index)

    clean_df = pd.concat([clean_df, lat_lon_poly_df], axis=1)

    # Encode categorical columns using One-Hot Encoding
    # If no encoder is provided, fit one on the current data
    if encoder is None:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoder.fit(clean_df[['CALLTYPE_CODE', 'neighborhood']])
    
    # Transform the categorical columns
    encoded_columns = encoder.transform(clean_df[['CALLTYPE_CODE', 'neighborhood']])

    # Create a DataFrame for the encoded columns and concatenate with the original data
    encoded_df = pd.DataFrame(encoded_columns, 
                              columns=encoder.get_feature_names_out(['CALLTYPE_CODE', 'neighborhood']), 
                              index=clean_df.index)

    # Concatenate encoded columns with the clean_df
    final_df = pd.concat([clean_df, encoded_df], axis=1)

    # Drop the original categorical columns since they have been encoded
    final_df = final_df.drop(columns=['CALLTYPE_CODE', 'neighborhood'])
    
    return final_df, encoder

def split_data(df, mapping_dict):
    # Define features and target
    X = df.drop(columns=['DISPO_SUBSET'])
    y = df['DISPO_SUBSET']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f'classes = {label_encoder.classes_}')

    label_names = [mapping_dict[code] for code in label_encoder.classes_]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print(X_train.shape)
    print(X_test.shape)

    class_labels = [mapping_dict[code] for code in label_encoder.inverse_transform([0, 1, 2, 3])]
    print(class_labels)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled data back to DataFrame, using the original column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    sample_sizes = Counter(y_train)
    print(sample_sizes)  # This will show you the current class distribution

    # make the majority class 50% of the data
    sample_sizes[3] = sum(sorted(list(sample_sizes.values()))[0:3])
    print(sample_sizes)

    # Undersample
    rus = RandomUnderSampler(sampling_strategy=sample_sizes, random_state=42)  
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled_df, y_train)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, label_names, class_labels, X_train_resampled, y_train_resampled

if __name__ == "__main__":
    print("Hello, world")