import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'food_inventory.csv')
PROCESSED_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'inventory_clean.csv')

def load_and_clean():

    #Step 1:Read raw CSV file into a DataFrame
    df = pd.read_csv(RAW_PATH)

    #Step 2:Convert purchase_date from string to datetime format
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    # Step 3: Convert expiry_date from string to datetime format
    df['expiry_date'] = pd.to_datetime(df['expiry_date'])

    # Step 4: Encode wasted column — yes → 1, no → 0
    df['wasted'] = df['wasted'].map({"yes": 1, "no": 0})

    # Step 5: Calculate shelf life in days
    df['shelf_life_days'] = (df['expiry_date'] - df['purchase_date']).dt.days

    # Step 6: Return cleaned DataFrame
    return df

if __name__ == "__main__":
    # Load and clean the data
    df = load_and_clean()
    # Save cleaned data to processed folder
    df.to_csv(PROCESSED_PATH, index=False)
    print("Data cleaned successfully!")
    print(df.head())     