import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# 1. Load cleaned dataset
df = pd.read_csv("dataset/movies_cleaned.csv")

# 2. Encode categorical feature (genre)
genre_encoder = LabelEncoder()
df['genre'] = genre_encoder.fit_transform(df['genre'])

# 3. Features & target
X = df[['domestic_revenue', 'foreign_revenue', 'genre', 'release_year']]
y = df['worldwide_revenue']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("✅ Model Training Completed")
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.2f}")

# 8. Save model & encoder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/box_office_model.pkl")
joblib.dump(genre_encoder, "model/genre_encoder.pkl")

print("📁 Model saved in 'model/' folder")
