#print_vectorizer.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

# Load data from CSV files
cleaned_data = pd.read_csv('cleaneddata.csv')
uncleaned_data = pd.read_csv('spammail.csv')

# Extract text columns and handle missing values
cleaned_texts = cleaned_data['Masseges'].fillna('').values
uncleaned_texts = uncleaned_data['Masseges'].fillna('').values

# Load the saved CountVectorizer
with open('best_vectorizer_logistic_regression.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Define specific row indices (0-based index, so actual rows are 10, 11, 12, 33)
selected_indices = [9, 10, 11, 32]  # Adjusted for 0-based indexing

# Print the selected messages and their transformations
for i, idx in enumerate(selected_indices):
    # Ensure the index is within the range of the dataset
    if idx >= len(uncleaned_texts) or idx >= len(cleaned_texts):
        print(f"Index {idx} is out of range. Skipping...\n")
        continue

    original_message = uncleaned_texts[idx]
    cleaned_message = cleaned_texts[idx]

    # Print the original message before cleaning
    print(f"Original Message {i+1}:\n{original_message}\n")

    # Print the cleaned message
    print(f"Cleaned Message {i+1}:\n{cleaned_message}\n")

    # Transform the cleaned message using the CountVectorizer
    message_vectorized = vectorizer.transform([cleaned_message])
    message_vectorized_dense = message_vectorized.toarray()

    # Print the vectorized message
    vectorized_df = pd.DataFrame(message_vectorized_dense, columns=vectorizer.get_feature_names_out())
    vectorized_df_filtered = vectorized_df.loc[:, (vectorized_df != 0).any(axis=0)]
    print(f"Vectorized Message (CountVectorizer) {i+1}:")
    print(vectorized_df_filtered.to_string(index=False))
    print("\n")

    # Convert the vectorized message to tf-idf representation
    tfidf_transformer = TfidfTransformer()
    message_tfidf = tfidf_transformer.fit_transform(message_vectorized)
    message_tfidf_dense = message_tfidf.toarray()

    # Print the tf-idf transformed message
    tfidf_df = pd.DataFrame(message_tfidf_dense, columns=vectorizer.get_feature_names_out())
    tfidf_df_filtered = tfidf_df.loc[:, (tfidf_df != 0).any(axis=0)]
    print(f"Vectorized Message (TF-IDF) {i+1}:")
    print(tfidf_df_filtered.to_string(index=False))
    print("\n")

    # Example: Print how the vectorized message is used in the model (assuming logistic regression)
    with open('best_model_logistic_regression.pkl', 'rb') as file:
        model = pickle.load(file)

    # Use the model to predict or transform (example: predicting the category)
    prediction = model.predict(message_tfidf)
    print(f"Model Prediction for Message {i+1}: {prediction[0]}")
    print("\n" + "="*50 + "\n")
