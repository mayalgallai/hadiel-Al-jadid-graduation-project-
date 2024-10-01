#loss_function.py
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import pandas as pd
from transformers import BertForSequenceClassification

# Load your data
data = pd.read_csv('cleaneddata.csv')
# Handle missing values in the 'Masseges' column
data['Masseges'].fillna('', inplace=True)
X = data['Masseges']
y = data['Category']
# Initialize CountVectorizer
vectorizer = CountVectorizer()
# Fit and transform the text data
X_transformed = vectorizer.fit_transform(X)

# Save the vectorizer for future use
with open('best_vectorizer_logistic_regression.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
# Initialize DecisionTreeClassifier
model =LogisticRegression()
  # زيادة عدد التكرارات
# Fit the model
model.fit(X_transformed, y)

# Save the model
with open('best_model_logistic_regression.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully.")

# Predict using the trained model
y_pred = model.predict(X_transformed)

# Calculate log loss if applicable
if hasattr(model, 'predict_proba'):
    loss = log_loss(y, model.predict_proba(X_transformed))
    print("Log Loss:", loss)
else:
    print("Model doesn't support probability predictions.")
