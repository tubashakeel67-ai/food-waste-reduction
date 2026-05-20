import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


# STEP 2: Load & Merge Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:", BASE_DIR)
train_df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'raw', 'train.csv'))
meal_df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'raw', 'meal_info.csv'))
print("Train Shape:", train_df.shape)
print("Meal Shape:", meal_df.shape)
df = train_df.merge(meal_df, on='meal_id')
print("Merged Shape:", df.shape)


# STEP 3: Engineer New Feature
df['discount'] = df['base_price'] - df['checkout_price']


# STEP 4: Select Features — base_price 
features = ['week', 'checkout_price',
            'discount',
            'emailer_for_promotion',
            'homepage_featured',
            'category']

X = df[features]
X = pd.get_dummies(X, columns=['category'])
print("X Shape:", X.shape)


# STEP 5: Target
y = df['num_orders']


# STEP 6: Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# STEP 7: Create Model
model = XGBRegressor(n_estimators=100, random_state=42)
print("Model Ready!")


# STEP 8: Train Model
model.fit(X_train, y_train)
print("Model Trained!")


# STEP 9: Predictions
y_pred = model.predict(X_test)


# STEP 10: RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)


# STEP 11: Feature Importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df.to_string(index=False))


# STEP 12: Save Model
joblib.dump(model, os.path.join(BASE_DIR, '..', 'models', 'demand_model.pkl'))
print("\nModel Saved!")


"""
########################################################################
                        Summary
########################################################################
Model 2: XGBoost Regressor — Demand Prediction

Performance:
- RMSE: 238.09 — high due to skewed order data
- Improvement: log transformation on target variable can reduce RMSE significantly

Key Findings:
- category_Rice Bowl is the most critical feature (21.9% importance)
- category_Sandwich is second most important (18.2% importance)
- homepage_featured drives demand significantly (8.2% importance)
- emailer_for_promotion impacts order volume (7.6% importance)
- discount surprisingly has low importance (1.2%)

Business Actions:
- Prioritize Rice Bowl and Sandwich inventory — highest demand categories
- Use homepage featuring strategically to boost low-demand meals
- Send emailer promotions for slow-moving meal categories
- Reduce production of Biryani and Pasta to avoid over-preparation

Improvement Path:
- Apply log transformation on num_orders to handle skewed data
- Include center_info.csv features for location-based predictions
- Add seasonal and weather data for more accurate forecasting

"""