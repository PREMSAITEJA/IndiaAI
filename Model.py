import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# Load the saved Random Forest model
rf_model = joblib.load('random_forest_model.pkl')

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the CSV file
file_path = 'test.csv'
text_column_name = 'crimeaditionalinfo'
label_column_name = 'category'  # Replace with your actual label column name in test.csv

# Load data from CSV and drop rows with NaN or empty text values
df = pd.read_csv(file_path)
df = df.dropna(subset=[text_column_name])
df = df[df[text_column_name].str.strip() != ""]

# Extract the text data and true labels
text_data = df[text_column_name].tolist()
# true_labels = df[label_column_name].tolist()
true_labels = [str(label).lower() for label in df[label_column_name].tolist()]  # Convert true labels to lowercase

# Vectorize the text data
text_data_vectorized = vectorizer.transform(text_data)

# Predict categories
# predicted_labels = rf_model.predict(text_data_vectorized)
predicted_labels = [str(label).lower() for label in rf_model.predict(text_data_vectorized)]  # Convert predicted labels to lowercase

# print(predicted_labels)

for i in range(0,10):
    print(true_labels[i] + " :" +predicted_labels[i] )

# Print each prediction for reference
# for i, (text, prediction) in enumerate(zip(text_data, predicted_labels), start=1):
    # print(f"{i}. Text: {text}\n   Predicted Category: {prediction}\n")

# Calculate accuracy and generate a classification report
accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels,zero_division=0)

# Print the accuracy and classification report
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Save the accuracy and report to a text file
with open("accuracy_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Accuracy and classification report saved to accuracy_report.txt.")