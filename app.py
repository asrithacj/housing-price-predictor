import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model
clf = pickle.load(open("classifier.pkl", "rb"))


st.title("🏡 India Housing Price Prediction Dashboard")
st.write("Enter the property details below:")

# ---------------------------------------------
# INPUT FIELDS (matching EXACT CSV column names)
# ---------------------------------------------
BHK = st.number_input("BHK", min_value=1, max_value=10, value=2)
Size_in_SqFt = st.number_input("Size in SqFt", min_value=100, max_value=10000, value=1000)
Price_in_Lakhs = st.number_input("Current Price in Lakhs", min_value=1, max_value=10000, value=50)
Price_per_SqFt = st.number_input("Price per SqFt", min_value=100, max_value=20000, value=5000)
Nearby_Schools = st.number_input("Nearby Schools", min_value=0, max_value=20, value=3)
Nearby_Hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=20, value=2)

# Categorical columns (must match training EXACTLY)
State = st.text_input("State")
City = st.text_input("City")
Locality = st.text_input("Locality")
Property_Type = st.text_input("Property Type")
Furnished_Status = st.text_input("Furnished Status")
Security = st.text_input("Security")
Amenities = st.text_input("Amenities")
Facing = st.text_input("Facing")
Owner_Type = st.text_input("Owner Type")
Availability_Status = st.text_input("Availability Status")
Public_Transport_Accessibility = st.text_input("Public Transport Accessibility")

# -------------------------
# Create input DataFrame
# -------------------------
input_dict = {
    "BHK": [BHK],
    "Size_in_SqFt": [Size_in_SqFt],
    "Price_in_Lakhs": [Price_in_Lakhs],
    "Price_per_SqFt": [Price_per_SqFt],
    "Nearby_Schools": [Nearby_Schools],
    "Nearby_Hospitals": [Nearby_Hospitals],
    "State": [State],
    "City": [City],
    "Locality": [Locality],
    "Property_Type": [Property_Type],
    "Furnished_Status": [Furnished_Status],
    "Security": [Security],
    "Amenities": [Amenities],
    "Facing": [Facing],
    "Owner_Type": [Owner_Type],
    "Availability_Status": [Availability_Status],
    "Public_Transport_Accessibility": [Public_Transport_Accessibility]
}

df = pd.DataFrame(input_dict)

# Label encoding for categorical fields (simple method)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype('category').cat.codes

# -------------------------
# Predictions
# -------------------------
if st.button("Predict"):
    invest_pred = clf.predict(df)[0]


    st.subheader("📌 Prediction Results")
    # Future Price Calculation
price_future = predicted_price * 1.35   # (Assuming 35% growth in 5 years)

    st.write(f"**Future Price in 5 Years:** ₹ {price_future:.2f} Lakhs")

    if invest_pred == 1:
        st.success("✔ Recommended as Good Investment")
    else:
        st.error("❌ Not Recommended as Good Investment")

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# FUTURE PRICE GROWTH GRAPH
# -------------------------
st.header("📈 Future Price Growth Projection")

years = [0, 1, 2, 3, 4, 5]
growth_rate = 0.08  # 8% per year
base_price = Price_in_Lakhs  # current price from user input

projected_prices = [base_price * ((1 + growth_rate) ** y) for y in years]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(years, projected_prices, marker='o')
ax.set_xlabel("Years")
ax.set_ylabel("Projected Price (Lakhs)")
ax.set_title("5-Year Property Price Projection")
st.pyplot(fig)

# -------------------------
# NEARBY FACILITIES GRAPH
# -------------------------
st.header("🏙 Nearby Facility Comparison")

facilities = ["Schools", "Hospitals"]
values = [Nearby_Schools, Nearby_Hospitals]

fig2, ax2 = plt.subplots()
ax2.bar(facilities, values, color=["blue", "green"])
ax2.set_ylabel("Count")
ax2.set_title("Nearby Schools & Hospitals")
st.pyplot(fig2)

