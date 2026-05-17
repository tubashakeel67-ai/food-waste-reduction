import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# Import data loading and cleaning function from eda module
from eda import load_and_clean

# Step 1: Load and clean the data
df = load_and_clean()

# Step 2: Select features and apply One-Hot Encoding on category column
X = df[['quantity', 'shelf_life_days', 'category']]
X = pd.get_dummies(X, columns=['category'])

# Step 3: Select target variable
y = df['wasted']

# Step 4: Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create Random Forest Classifier with 100 trees
model = RandomForestClassifier(n_estimators=100)

# Step 6: Train the model on training data
model.fit(X_train, y_train)

# Step 7: Make predictions on test data
y_pred = model.predict(X_test)

# Step 8: Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8b: Print feature importance with names
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance with Names:")
print(importance_df.to_string(index=False))

# Step 9: Save trained model to models folder
joblib.dump(model, 'models/waste_model.pkl')
print("✅ Model saved successfully!")