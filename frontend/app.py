import os
import streamlit as st
import requests
import pandas as pd

# STEP 2: Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INVENTORY_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'inventory_clean.csv')
API_URL = "http://localhost:8000"

# STEP 3: Page config
st.set_page_config(
    page_title="AI Food Waste System",
    page_icon="🍱",
    layout="wide"
)

# STEP 4: Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Page", [
    "Inventory Overview",
    "Expiry Alerts",
    "Waste Predictor",
    "Demand Forecaster"
])

# STEP 5: PAGE 1 — Inventory Overview
if page == "Inventory Overview":
    st.title("📦 Inventory Overview")
    df = pd.read_csv(INVENTORY_PATH)
    st.dataframe(df)

# STEP 6: PAGE 2 — Expiry Alerts
elif page == "Expiry Alerts":
    st.title("🚨 Expiry Alerts")
    try:
        response = requests.get(f"{API_URL}/inventory/alerts")
        alerts = response.json()
        st.warning(f"⚠️ {len(alerts['alerts'])} items expire soon!")
        st.error("These items need immediate attention!")
        alerts_df = pd.DataFrame(alerts['alerts'])

        def highlight_row(row):
            if row['shelf_life_days'] <= 2:
                return ['background-color: red'] * len(row)
            elif row['shelf_life_days'] <= 5:
                return ['background-color: #ffffcc'] * len(row)
            else:
                return ['background-color: white'] * len(row)

        styled_df = alerts_df.style.apply(highlight_row, axis=1)
        st.dataframe(styled_df)
    except:
        st.error("❌ API not running! Please start FastAPI first!")

# STEP 7: PAGE 3 — Waste Predictor
elif page == "Waste Predictor":
    st.title("🗑️ Waste Predictor")
    category = st.selectbox("Select Category", [
        'Dairy', 'Bakery', 'Vegetables', 'Meat', 'Fruits'
    ])
    quantity = st.number_input("Enter Quantity")
    shelf_life = st.number_input("Enter Shelf Life Days")

    if st.button("Predict!"):
        try:
            response = requests.post(
                f"{API_URL}/predict/expiry",
                json={
                    "quantity": int(quantity),
                    "shelf_life_days": int(shelf_life),
                    "category": category
                }
            )
            result = response.json()
            if result['wasted'] == 1:
                st.error("⚠️ HIGH WASTE RISK!")
            else:
                st.success("✅ LOW WASTE RISK!")
        except:
            st.error("❌ API not running! Please start FastAPI first!")

# STEP 8: PAGE 4 — Demand Forecaster
elif page == "Demand Forecaster":
    st.title("📈 Demand Forecaster")
    week = st.number_input("Enter Week")
    checkout_price = st.number_input("Enter Checkout Price")
    discount = st.number_input("Enter Discount")
    emailer = st.selectbox("Emailer for Promotion", [0, 1])
    homepage = st.selectbox("Homepage Featured", [0, 1])
    category = st.selectbox("Select Category", [
        'Beverages', 'Biryani', 'Desert',
        'Extras', 'Fish', 'Other Snacks',
        'Pasta', 'Pizza', 'Rice Bowl',
        'Salad', 'Sandwich', 'Seafood',
        'Soup', 'Starters'
    ])

    if st.button("Forecast!"):
        try:
            response = requests.post(
                f"{API_URL}/predict/demand",
                json={
                    "week": int(week),
                    "checkout_price": float(checkout_price),
                    "discount": float(discount),
                    "emailer_for_promotion": int(emailer),
                    "homepage_featured": int(homepage),
                    "category": category
                }
            )
            result = response.json()
            st.success(f"📦 Predicted Orders: {result['predicted_orders']}")
        except:
            st.error("❌ API not running! Please start FastAPI first!")