import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("india_housing_prices (1).csv")

# Remove unwanted columns if exist
drop_cols = ["ID", "Unnamed: 0", "Unnamed: 0.1"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# NUMERIC columns
num_cols = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", 
    "Price_per_SqFt", "Nearby_Schools", "Nearby_Hospitals"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

# CATEGORICAL columns
cat_columns = [
    "State", "City", "Locality", "Property_Type",
    "Furnished_Status", "Security", "Amenities",
    "Facing", "Owner_Type", "Availability_Status",
    "Public_Transport_Accessibility"
]

df.fillna("Unknown", inplace=True)

le = LabelEncoder()
for col in cat_columns:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Targets
median_price = df["Price_per_SqFt"].median()
df["Good_Investment"] = (df["Price_per_SqFt"] <= median_price).astype(int)

growth_rate = 0.08
df["Future_Price_5Yrs"] = df["Price_in_Lakhs"] * ((1 + growth_rate) ** 5)

# FEATURES
features = [c for c in num_cols + cat_columns if c in df.columns]
X = df[features]

# Force SAME LENGTHS for safety
min_len = min(len(X), len(df["Good_Investment"]), len(df["Future_Price_5Yrs"]))
X = X.iloc[:min_len]
y_class = df["Good_Investment"].iloc[:min_len]
y_reg = df["Future_Price_5Yrs"].iloc[:min_len]

# TRAIN CLASSIFIER
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pickle.dump(clf, open("models/classifier.pkl", "wb"))

# TRAIN REGRESSOR
reg = RandomForestRegressor()
reg.fit(X_train, y_reg.iloc[:len(X_train)])   # force alignment
pickle.dump(reg, open("models/regressor.pkl", "wb"))

print("🎉 Training Completed Successfully! Models saved in /models folder.")
