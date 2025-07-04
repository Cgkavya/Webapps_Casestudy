import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("beer-servings.csv", index_col=0)

# Handle missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Features and target
X = df[["country", "continent", "beer_servings", "spirit_servings", "wine_servings"]]
y = df["total_litres_of_pure_alcohol"]

# OneHot Encoding
categorical_cols = ["country", "continent"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(
    encoded, columns=encoder.get_feature_names_out(categorical_cols)
)

X_encoded = pd.concat([encoded_df, X.drop(columns=categorical_cols)], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Model : Linear Regression

lr = LinearRegression()
model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)


print("R2 Score:", r2_score(y_test, predictions))
print("Mean square Score:", mean_squared_error(y_test, predictions))

alphas = [0.01, 0.1, 1, 10, 100]


lasso = Lasso(max_iter=10000)
lasso_params = {"alpha": alphas}

lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring="r2")
lasso_grid.fit(X_train, y_train)

lasso_best = lasso_grid.best_estimator_
lasso_preds = lasso_best.predict(X_test)

print("🔹 Lasso Best Alpha:", lasso_grid.best_params_["alpha"])
print("🔹 Lasso Test R2 Score:", r2_score(y_test, lasso_preds))
print("🔹 Lasso Test MSE:", mean_squared_error(y_test, lasso_preds))

ridge = Ridge()
ridge_params = {"alpha": alphas}

ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring="r2")
ridge_grid.fit(X_train, y_train)

ridge_best = ridge_grid.best_estimator_
ridge_preds = ridge_best.predict(X_test)

print("🔹 Ridge Best Alpha:", ridge_grid.best_params_["alpha"])
print("🔹 Ridge Test R2 Score:", r2_score(y_test, ridge_preds))
print("🔹 Ridge Test MSE:", mean_squared_error(y_test, ridge_preds))


# Model: Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))


param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best R2 Score on CV:", grid.best_score_)

# Predict using the best estimator
best_model = grid.best_estimator_
preds_best = best_model.predict(X_test)
print("Test R2 Score with Best Model:", r2_score(y_test, preds_best))

models = {
    "Random Forest": r2_score(y_test, preds),
    "Ridge": r2_score(y_test, ridge_preds),
    "Lasso": r2_score(y_test, lasso_preds),
}

# Print results
for name, score in models.items():
    print(f"{name} R² Score: {score}")

# Choose the best model
best_model_name = max(models, key=models.get)
print("✅ Best Model:", best_model_name)


# Save the best model to a .pkl file
if best_model_name == "Random Forest":
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model, f)
elif best_model_name == "Ridge":
    with open("best_model.pkl", "wb") as f:
        pickle.dump(ridge_best, f)
elif best_model_name == "Lasso":
    with open("best_model.pkl", "wb") as f:
        pickle.dump(lasso_best, f)

print("✅ Best model saved as 'best_model.pkl' using pickle.")

# Save the best model and encoder + feature names
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)  # Use best_model from GridSearch

# Save encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# Save the exact column order used for training
with open("model_features.pkl", "wb") as f:
    pickle.dump(X_encoded.columns.tolist(), f)
