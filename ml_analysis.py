import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sqlite3
import joblib

def load_data_from_sql():
    conn = sqlite3.connect('f1.db')
    
    # Load relevant data for prediction
    query = """
    SELECT 
        r.raceId,
        r.year,
        r.round,
        c.circuitId,
        c.name as circuit_name,
        c.country,
        c.lat,
        c.lng,
        c.alt,
        res.position,
        res.points,
        res.grid,
        res.laps,
        res.milliseconds,
        res.fastestLap,
        res.fastestLapTime,
        res.fastestLapSpeed,
        d.driverId,
        d.nationality as driver_nationality,
        con.constructorId,
        con.nationality as constructor_nationality
    FROM races r
    JOIN circuits c ON r.circuitId = c.circuitId
    JOIN results res ON r.raceId = res.raceId
    JOIN drivers d ON res.driverId = d.driverId
    JOIN constructors con ON res.constructorId = con.constructorId
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    # Handle NULL values in fastestLapTime
    df['fastestLapTime'] = df['fastestLapTime'].replace('\\N', np.nan)
    
    # Convert time strings to seconds for non-null values
    def convert_time(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            time_parts = time_str.split(':')
            if len(time_parts) == 3:
                minutes, seconds, milliseconds = time_parts
                return float(minutes) * 60 + float(seconds) + float(milliseconds) / 1000
            return np.nan
        except:
            return np.nan
    
    df['fastestLapTime'] = df['fastestLapTime'].apply(convert_time)
    
    # Convert numeric columns, replacing \N with NaN
    numeric_columns = ['fastestLapSpeed', 'fastestLap', 'grid', 'laps']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace('\\N', np.nan), errors='coerce')
    
    # Fill missing values with median for numeric columns
    for col in numeric_columns + ['fastestLapTime']:
        df[col] = df[col].fillna(df[col].median())
    
    # Create features
    features = ['year', 'round', 'circuitId', 'grid', 'laps', 
                'fastestLap', 'fastestLapTime', 'fastestLapSpeed',
                'lat', 'lng', 'alt']
    
    # Ensure all feature columns are numeric
    X = df[features].apply(pd.to_numeric, errors='coerce')
    
    # Fill any remaining NaN values with column medians
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    y = pd.to_numeric(df['position'], errors='coerce').fillna(-1)  # Use -1 for unknown positions
    
    return X, y

def train_model():
    # Load and preprocess data
    print("Loading data from SQL...")
    df = load_data_from_sql()
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    print("Splitting data...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    print("Saving model...")
    # Save model and scaler
    joblib.dump(model, 'f1_model.joblib')
    joblib.dump(scaler, 'f1_scaler.joblib')
    
    return model, scaler

if __name__ == "__main__":
    train_model() 