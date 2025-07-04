import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data for visualization
df = pd.read_csv("beer-servings.csv", index_col=0)
df.fillna(df.mode().iloc[0], inplace=True)

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoder
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load model feature names
with open("model_features.pkl", "rb") as f:
    model_features = pickle.load(f)

# Define categorical columns used during training
categorical_cols = ["country", "continent"]

# ----------------------------
# 🚀 Landing Page - Infographics
# ----------------------------
st.set_page_config(page_title="Alcohol Predictor Dashboard", layout="centered")

st.title("🍻 Alcohol Consumption Dashboard")

st.markdown("## 🌍 Average Alcohol Servings by Continent")

avg_continent = df.groupby("continent")[
    ["beer_servings", "spirit_servings", "wine_servings"]
].mean()

fig, ax = plt.subplots(figsize=(10, 5))
avg_continent.plot(kind="bar", ax=ax)
plt.title("Average Servings by Continent")
plt.ylabel("Average Servings")
st.pyplot(fig)

# Pie chart of dataset by continent
st.markdown("## 🧭 Distribution by Continent")
continent_counts = df["continent"].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(
    continent_counts, labels=continent_counts.index, autopct="%1.1f%%", startangle=90
)
ax2.axis("equal")
st.pyplot(fig2)

st.markdown("---")

# ----------------------------
# 📊 Prediction Section
# ----------------------------
st.header("🔮 Predict Pure Alcohol Consumption")

# Dropdown for categorical variable
country = st.selectbox("Select Country", sorted(df["country"].unique().tolist()))
continent = st.selectbox(
    "Select Continent", sorted(df["continent"].dropna().unique().tolist())
)

# Numerical inputs
beer_servings = st.number_input("Beer Servings", min_value=0, step=1)
spirit_servings = st.number_input("Spirit Servings", min_value=0, step=1)
wine_servings = st.number_input("Wine Servings", min_value=0, step=1)

# Prepare input
input_data = {
    "country": country,
    "continent": continent,
    "beer_servings": beer_servings,
    "spirit_servings": spirit_servings,
    "wine_servings": wine_servings,
}
input_df = pd.DataFrame([input_data])

# Encode categorical features
encoded = encoder.transform(input_df[categorical_cols])
encoded_df = pd.DataFrame(
    encoded, columns=encoder.get_feature_names_out(categorical_cols)
)

# Merge with numerical
input_encoded = pd.concat(
    [input_df.drop(columns=categorical_cols).reset_index(drop=True), encoded_df], axis=1
)

# Ensure all model features are present
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder
input_encoded = input_encoded[model_features]

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Estimated Total Litres of Pure Alcohol: **{prediction:.2f}**")
