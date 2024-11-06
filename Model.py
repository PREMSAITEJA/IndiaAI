import joblib
import pandas as pd

# Load the saved Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example: Predict on new data
new_data = ["I have ordered  saree and  blouse from rinki sur I paid all amount   after take payment she didnt send me my saree and blouse After  months she refund me  only after that I dont recive my  everyday I told him about my money but she didnt refund and ignore me"]
new_data_vectorized = vectorizer.transform(new_data)
predicted_category = rf_model.predict(new_data_vectorized)

print("Predicted Category:", predicted_category)
