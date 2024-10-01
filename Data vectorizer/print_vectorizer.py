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

# Train and save CountVectorizer and TfidfTransformer
def train_and_save_models(cleaned_texts):
    # Initialize and fit the CountVectorizer
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(cleaned_texts)

    # Initialize and fit the TfidfTransformer
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(X_counts)

    # Save the CountVectorizer
    with open('best_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)

    # Save the TfidfTransformer
    with open('best_tfidf_transformer.pkl', 'wb') as file:
        pickle.dump(tfidf_transformer, file)

    print("CountVectorizer and TfidfTransformer have been trained and saved.")

# Uncomment this to train and save models (do this only once)
# train_and_save_models(cleaned_texts)

# Load the saved CountVectorizer and TfidfTransformer
with open('best_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('best_tfidf_transformer.pkl', 'rb') as file:
    tfidf_transformer = pickle.load(file)

# Define specific row indices (0-based index, so actual rows are 10, 11, 12, 33)
selected_indices = [9, 10, 11, 32]  # Adjusted for 0-based indexing

# Open a file to write the results
with open('results.txt', 'w') as file:
    # Print the selected messages and their transformations
    for i, idx in enumerate(selected_indices):
        # Ensure the index is within the range of the dataset
        if idx >= len(uncleaned_texts) or idx >= len(cleaned_texts):
            file.write(f"Index {idx} is out of range. Skipping...\n\n")
            continue

        original_message = uncleaned_texts[idx]
        cleaned_message = cleaned_texts[idx]

        # Write the original message before cleaning
        file.write(f"Original Message {i+1}:\n{original_message}\n\n")

        # Write the cleaned message
        file.write(f"Cleaned Message {i+1}:\n{cleaned_message}\n\n")

        # Transform the cleaned message using the saved CountVectorizer
        message_vectorized = vectorizer.transform([cleaned_message])
        message_vectorized_dense = message_vectorized.toarray()

        # Calculate Bag of Words
        bow_df = pd.DataFrame(message_vectorized_dense, columns=vectorizer.get_feature_names_out())
        bow_df_filtered = bow_df.loc[:, (bow_df != 0).any(axis=0)]
        file.write(f"Bag of Words (BoW) for Message {i+1}:\n")
        file.write(bow_df_filtered.to_string(index=False))
        file.write("\n\n")

        # Calculate TF
        tf = message_vectorized_dense / message_vectorized_dense.sum(axis=1, keepdims=True)
        tf_df = pd.DataFrame(tf, columns=vectorizer.get_feature_names_out())
        tf_df_filtered = tf_df.loc[:, (tf_df != 0).any(axis=0)]
        file.write(f"Term Frequency (TF) for Message {i+1}:\n")
        file.write(tf_df_filtered.to_string(index=False))
        file.write("\n\n")

        # Calculate IDF for the entire corpus
        idf = tfidf_transformer.idf_
        idf_df = pd.DataFrame(idf.reshape(1, -1), columns=vectorizer.get_feature_names_out())
        
        # Filter IDF values to only include words present in the message
        words_in_message = tf_df_filtered.columns
        idf_df_filtered = idf_df[words_in_message]
        
        file.write(f"Inverse Document Frequency (IDF) for Words in Message {i+1}:\n")
        file.write(idf_df_filtered.to_string(index=False))
        file.write("\n\n")

        # Calculate TF-IDF
        message_tfidf = tfidf_transformer.transform(message_vectorized)
        message_tfidf_dense = message_tfidf.toarray()
        tfidf_df = pd.DataFrame(message_tfidf_dense, columns=vectorizer.get_feature_names_out())
        tfidf_df_filtered = tfidf_df.loc[:, (tfidf_df != 0).any(axis=0)]
        file.write(f"Vectorized Message (TF-IDF) {i+1}:\n")
        file.write(tfidf_df_filtered.to_string(index=False))
        file.write("\n\n")

    
     