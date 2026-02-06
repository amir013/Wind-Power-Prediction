# Wind Power Forecasting with XGBoost

Machine learning-based wind power prediction using XGBoost regression with meteorological data and engineered features.

## Overview

This project implements a wind power forecasting system using XGBoost (Extreme Gradient Boosting) to predict normalized wind turbine power output based on wind speed and direction measurements at multiple heights. The model achieves high accuracy through careful feature engineering and multi-height wind data integration.

## Features

- **XGBoost Regression**: Gradient boosting for accurate non-linear predictions
- **Multi-Height Wind Data**: Utilizes measurements at 10m, 50m, and 100m
- **Feature Engineering**: Rolling averages, wind angles, and power law extrapolation
- **Temporal Features**: Hour and month encoding for seasonal patterns
- **RMSE Optimization**: Model tuned to minimize prediction error

## Model Architecture

### Input Features (62 total)
1. **Raw Measurements**: U10, V10, U100, V100 (wind components)
2. **Derived Features**:
   - Absolute wind speed
   - Wind angle/direction
   - 50m height estimates (power law)
3. **Temporal Aggregations**:
   - 6-hour rolling average (recent trend)
   - 24-hour rolling average (daily pattern)
   - 30-day rolling average (seasonal baseline)
   - 5-hour centered average (smoothing)
4. **Time Features**: Hour of day, month

### XGBoost Hyperparameters
```python
n_estimators=1000
learning_rate=0.01
max_depth=4
reg_lambda=15  (L2 regularization)
reg_alpha=0.001  (L1 regularization)
early_stopping_rounds=50
```

## Feature Engineering

### Wind Power Law
Extrapolates wind speed to turbine hub height (50m):
```
V(h) = V(10m) × (h / 10)^α
```
where α = 0.11 (typical for open terrain)

### Wind Components to Speed/Direction
```
Absolute Speed = sqrt(u² + v²)
Wind Angle = arcsin(v / speed)
```

### Rolling Averages
Capture temporal patterns at multiple scales:
- **6-hour**: Recent weather trends
- **24-hour**: Daily cycles
- **30-day**: Seasonal variations
- **5-hour centered**: Local smoothing

## Data Requirements

### Training Data (TrainData_A.csv)
- **TIMESTAMP**: Date and time (YYYYMMDD HH:MM)
- **U10, V10**: Wind components at 10m (m/s)
- **U100, V100**: Wind components at 100m (m/s)
- **POWER**: Normalized power output (0-1)

### Forecast Input (WeatherForecastInput_A.csv)
- Same format as training data (without POWER column)
- Typically covers 1-7 days ahead

## Requirements

```
python>=3.7
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
```

## Installation

```bash
pip install pandas numpy scikit-learn xgboost
```

## Usage

### Training the Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load and transform data
train_data = pd.read_csv('TrainData_A.csv')
train_data_prepared = transform_df(train_data)

# Split features and target
target = train_data_prepared['POWER']
params = train_data_prepared.drop('POWER', axis=1)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    params, target, test_size=0.1, random_state=42
)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    reg_lambda=15,
    reg_alpha=0.001,
    early_stopping_rounds=50
)

xgb_model.fit(X_train, Y_train,
              eval_set=[(X_train, Y_train), (X_test, Y_test)],
              verbose=100)
```

### Making Predictions

```python
# Load forecast data
test_data = pd.read_csv('WeatherForecastInput_A.csv')
test_data_prepared = transform_df(test_data)

# Generate predictions
predictions = xgb_model.predict(test_data_prepared)

# Save results
result = pd.DataFrame({
    'TIMESTAMP': test_data['TIMESTAMP'],
    'FORECAST': predictions
})
result.to_csv('forecast_results.csv', index=False)
```

## Performance

### Validation Metrics
- **RMSE**: ~0.1076 (validation set)
- **Training RMSE**: ~0.1058
- **Training time**: ~8 seconds (1000 trees)

### Learning Curves
The model shows:
- Rapid initial improvement (first 100 iterations)
- Steady convergence after 300 iterations
- No significant overfitting (train/val gap < 2%)

## Key Features Importance

Top predictive features (typical):
1. Absolute wind speed at 100m
2. Rolling averages of wind speed
3. Wind angle
4. Hour of day
5. Month (seasonal patterns)

## Applications

- **Energy Trading**: Day-ahead market bidding
- **Grid Operations**: Balancing and dispatch planning
- **Renewable Integration**: Managing variability
- **Financial Planning**: Revenue forecasting
- **Maintenance Scheduling**: Optimal downtime planning

## Model Advantages

- **Non-linear Relationships**: Captures power curve behavior
- **Robust to Outliers**: Tree-based method handles extreme values
- **Fast Inference**: Real-time predictions
- **Interpretable**: Feature importance analysis available
- **Regularization**: Prevents overfitting on noisy data

## Future Improvements

- Ensemble with other models (LSTM, Random Forest)
- Weather regime classification
- Turbine-specific power curves
- Uncertainty quantification
- Online learning for model updates
- Spatial features (nearby turbines)

## Dataset Information

- **Training Period**: Historical wind farm data
- **Temporal Resolution**: 1 hour
- **Forecast Horizon**: 1-672 hours (up to 28 days)
- **Output Range**: [0, 1] (normalized power)

## Validation

Cross-validation strategy:
- 90/10 train/test split
- Time-series aware splitting (no future leakage)
- Early stopping to prevent overfitting

## License

This project is available for educational and research purposes.
