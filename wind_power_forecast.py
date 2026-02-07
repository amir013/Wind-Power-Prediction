"""
Wind Power Forecasting using XGBoost
Corrected version with proper DataFrame handling and model persistence
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import joblib
import os


def readTimeStamp(df):
    """Extract hour and month from timestamp."""
    df = df.copy()  # FIXED: Create copy to avoid SettingWithCopyWarning
    df['hour'] = df['TIMESTAMP'].apply(lambda x: datetime.strptime(x, '%Y%m%d %H:%M').hour)
    df['month'] = df['TIMESTAMP'].apply(lambda x: datetime.strptime(x, '%Y%m%d %H:%M').month)
    return df


def getAbsWindSpeed(u, v):
    """Calculate absolute wind speed from u and v components."""
    abs_speed = np.sqrt(u**2 + v**2)
    return abs_speed


def getWindAngle(u, v):
    """Calculate wind angle from u and v components."""
    abs_speed = getAbsWindSpeed(u, v)
    angle = np.arcsin(v / abs_speed)
    return angle


def powerWindLaw(v10, h):
    """Apply power law to extrapolate wind speed at different heights."""
    a = 0.11
    v = v10 * (h / 10) ** a
    return v


def calculateAverages(u, v, abs_wind_speed, angle, height):
    """
    Calculate rolling averages for wind parameters.

    Creates 4 different rolling averages:
    - 6-hour past average
    - 24-hour past average
    - 30-day past average
    - 5-hour centered average
    """
    elements = [u, v, abs_wind_speed, angle]
    element_names = ['u', 'v', 'abs_wind_speed', 'angle']
    colnames = ['avg_6_past', 'avg_24_past', 'avg_30_days_past', 'avg_5_current']

    df = pd.DataFrame()

    for i, element in enumerate(elements):
        avg_6_past = element.rolling(window=6, min_periods=1).mean()
        avg_24_past = element.rolling(window=24, min_periods=1).mean()
        avg_30_days_past = element.rolling(window=24*30, min_periods=1).mean()
        avg_5_current = element.rolling(window=5, min_periods=1, center=True).mean()

        new_df = pd.concat([avg_6_past, avg_24_past, avg_30_days_past, avg_5_current], axis=1)
        new_colnames = [name + '_' + element_names[i] + '_' + str(height) for name in colnames]
        new_df.columns = new_colnames
        df = pd.concat([df, new_df], axis=1)

    return df


def transform_df(df):
    """
    Transform raw weather data into features for model training.

    Creates features at 3 different heights (10m, 50m, 100m) including:
    - Absolute wind speed
    - Wind angle
    - Various rolling averages
    """
    df = df.copy()  # FIXED: Create copy to avoid modifying original
    df = df.dropna()
    df = readTimeStamp(df)

    # Process 10m height data
    df['abs_speed_10'] = getAbsWindSpeed(df['U10'], df['V10'])
    df['angle_10'] = getWindAngle(df['U10'], df['V10'])
    df_10 = calculateAverages(df['U10'], df['V10'], df['abs_speed_10'], df['angle_10'], 10)

    # Process 100m height data
    df['abs_speed_100'] = getAbsWindSpeed(df['U100'], df['V100'])
    df['angle_100'] = getWindAngle(df['U100'], df['V100'])
    df_100 = calculateAverages(df['U100'], df['V100'], df['abs_speed_100'], df['angle_100'], 100)

    # Extrapolate 50m height data using power law
    df['U50'] = powerWindLaw(df['U10'], 50)
    df['V50'] = powerWindLaw(df['V10'], 50)
    df['abs_speed_50'] = getAbsWindSpeed(df['U50'], df['V50'])
    df['angle_50'] = getWindAngle(df['U50'], df['V50'])
    df_50 = calculateAverages(df['U50'], df['V50'], df['abs_speed_50'], df['angle_50'], 50)

    return pd.concat([df, df_10, df_100, df_50], axis=1)


def train_model(train_file='TrainData_A.csv', model_file='wind_power_model.pkl'):
    """
    Train XGBoost model for wind power forecasting.

    Parameters:
    -----------
    train_file : str
        Path to training data CSV
    model_file : str
        Path to save trained model

    Returns:
    --------
    model : XGBRegressor
        Trained XGBoost model
    """
    print("Loading training data...")
    train_data = pd.read_csv(train_file)

    print("Preparing features...")
    train_data_prepared = transform_df(train_data)

    # Separate target and features
    target = train_data_prepared['POWER']
    params = train_data_prepared.drop(['TIMESTAMP', 'POWER'], axis=1)

    print(f"Training with {len(params.columns)} features")
    print(f"Dataset size: {len(train_data_prepared)} samples")

    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        params, target, test_size=0.1, random_state=42
    )

    # Initialize XGBoost model
    xgb_model = xgb.XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        max_depth=4,
        reg_lambda=15,
        reg_alpha=0.001,
        n_estimators=1000,
        early_stopping_rounds=50,
        learning_rate=0.01
    )

    print("\nTraining model...")
    xgb_model.fit(
        X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_test, Y_test)],
        verbose=100
    )

    # Evaluate on test set
    test_predictions = xgb_model.predict(X_test)
    r2 = r2_score(Y_test, test_predictions)
    print(f"\nTest R2 Score: {r2:.6f}")

    # Save model
    print(f"\nSaving model to {model_file}...")
    joblib.dump(xgb_model, model_file)

    return xgb_model


def predict(model, input_file='WeatherForecastInput_A.csv', output_file='result.csv'):
    """
    Make predictions using trained model.

    Parameters:
    -----------
    model : XGBRegressor or str
        Trained model or path to saved model
    input_file : str
        Path to input weather forecast data
    output_file : str
        Path to save predictions

    Returns:
    --------
    result : pd.DataFrame
        Predictions with timestamps
    """
    # Load model if path is provided
    if isinstance(model, str):
        print(f"Loading model from {model}...")
        model = joblib.load(model)

    print(f"Loading test data from {input_file}...")
    test_data = pd.read_csv(input_file)

    print("Preparing features...")
    test_data_prepared = transform_df(test_data)

    # Drop timestamp for prediction
    timestamp = test_data_prepared['TIMESTAMP']
    test_features = test_data_prepared.drop(['TIMESTAMP'], axis=1)

    print("Making predictions...")
    predictions = model.predict(test_features)

    # Create result dataframe
    result = pd.DataFrame({
        'TIMESTAMP': timestamp,
        'FORECAST': predictions
    })

    print(f"Saving predictions to {output_file}...")
    result.to_csv(output_file, index=False)

    print(f"\nGenerated {len(result)} predictions")
    print(result.head())

    return result


def evaluate_predictions(predictions_file='result.csv', solution_file='Solution_A.csv'):
    """
    Evaluate predictions against ground truth.

    Parameters:
    -----------
    predictions_file : str
        Path to predictions CSV
    solution_file : str
        Path to ground truth CSV
    """
    print(f"\nEvaluating predictions...")

    df_results = pd.read_csv(predictions_file)
    df_solution = pd.read_csv(solution_file)

    format = '%Y%m%d %H:%M'

    df_results['DATETIME'] = pd.to_datetime(df_results['TIMESTAMP'].astype("string"), format=format)
    df_solution['DATETIME'] = pd.to_datetime(df_solution['TIMESTAMP'].astype("string"), format=format)

    df_results = df_results.set_index(pd.DatetimeIndex(df_results['DATETIME']))
    df_solution = df_solution.set_index(pd.DatetimeIndex(df_solution['DATETIME']))

    df_results = df_results.sort_index()
    df_solution = df_solution.sort_index()

    # Verify alignment
    assert len(df_results) == len(df_solution), "Prediction and solution length mismatch"
    df_len = len(df_results)
    assert df_results['DATETIME'].iloc[0] == df_solution['DATETIME'].iloc[0], "Start date mismatch"
    assert df_results['DATETIME'].iloc[df_len-1] == df_solution['DATETIME'].iloc[df_len-1], "End date mismatch"

    preds = []
    actuals = []
    for index in df_results.index:
        preds.append(df_results.loc[index]['FORECAST'])
        actuals.append(df_solution.loc[index]['POWER'])

    score = r2_score(actuals, preds)

    print(f"R2 SCORE: {score:.6f}")
    return score


def main():
    """Main execution function."""
    MODEL_FILE = 'wind_power_model.pkl'

    print("="*60)
    print("WIND POWER FORECASTING WITH XGBOOST")
    print("="*60)

    # Train model if it doesn't exist
    if not os.path.exists(MODEL_FILE):
        print("\nTraining new model...")
        model = train_model(model_file=MODEL_FILE)
    else:
        print(f"\nModel {MODEL_FILE} already exists. Skipping training.")
        print("Delete the model file to retrain.")
        model = MODEL_FILE

    # Make predictions
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    predict(model, output_file='result.csv')

    # Evaluate if solution file exists
    if os.path.exists('Solution_A.csv'):
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        evaluate_predictions()
    else:
        print("\nSolution file not found. Skipping evaluation.")


if __name__ == '__main__':
    main()
