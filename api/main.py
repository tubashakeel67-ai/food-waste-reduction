import os
import pandas as pd
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Base directory for correct file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create FastAPI app instance
app = FastAPI(
    title="Food Waste API",
    description="Predict food waste and demand",
    version="1.0"
)

# Step 3: Load both models at startup
waste_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'waste_model.pkl'))
demand_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'demand_model.pkl'))

# Training columns for waste model
WASTE_CATEGORIES = [
    'quantity', 'shelf_life_days',
    'category_Bakery', 'category_Beverages', 'category_Canned',
    'category_Dairy', 'category_Fruit', 'category_Grains',
    'category_Meat', 'category_Vegetables'
]

# Training columns for demand model
DEMAND_CATEGORIES = [
    'week', 'checkout_price', 'discount',
    'emailer_for_promotion', 'homepage_featured','category_Beverages',
    'category_Biryani','category_Desert','category_Extras','category_Fish',
    'category_Other Snacks','category_Pasta','category_Pizza','category_Rice Bowl',
    'category_Salad','category_Sandwich','category_Seafood','category_Soup',
    'category_Starters'
]

# Step 4: Define input schemas using Pydantic
class WasteInput(BaseModel):
    quantity: int
    shelf_life_days: int
    category: str

class DemandInput(BaseModel):
    week: int
    checkout_price: float
    discount: float
    emailer_for_promotion: int
    homepage_featured: int
    category: str

# Step 5: Endpoint 1 — Predict food waste
@app.post("/predict/expiry")
def predict_expiry(data: WasteInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([{
        'quantity': data.quantity,
        'shelf_life_days': data.shelf_life_days,
        'category': data.category
    }])

    # Apply One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=['category'])

    # Add missing columns as 0
    for col in WASTE_CATEGORIES:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only training columns in correct order
    input_df = input_df[WASTE_CATEGORIES]

    # Make prediction
    prediction = waste_model.predict(input_df)
    return {"wasted": int(prediction[0])}

# Step 6: Endpoint 2 — Predict meal demand
@app.post("/predict/demand")
def predict_demand(data: DemandInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([{
        'week': data.week,
        'checkout_price': data.checkout_price,
        'discount': data.discount,
        'emailer_for_promotion': data.emailer_for_promotion,
        'homepage_featured': data.homepage_featured,
        'category': data.category
    }])

    # Apply One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=['category'])

    # Add missing columns as 0
    for col in DEMAND_CATEGORIES:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only training columns in correct order
    input_df = input_df[DEMAND_CATEGORIES]

    # Make prediction
    prediction = demand_model.predict(input_df)
    return {"predicted_orders": int(prediction[0])}

# Step 7: Endpoint 3 — Inventory alerts
@app.get("/inventory/alerts")
def inventory_alerts():
    # Load cleaned inventory data
    df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'processed', 'inventory_clean.csv'))

    # Filter items expiring within 7 days
    alerts = df[df['shelf_life_days'] < 7]

    return {"alerts": alerts.to_dict(orient='records')}