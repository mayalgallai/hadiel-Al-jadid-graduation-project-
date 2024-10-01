# May-Moktar-Algallai\hadiel-Al-jadidi-graduation-project-
preprocessing\project.ipynb
Data Cleaning
In this project, text data is cleaned through several steps to ensure better quality for machine learning models:

Lowercasing Text: All text is converted to lowercase to ensure consistency.

Replacing URLs: All URLs are replaced with the word "URL" to avoid confusion from web links.

Replacing Currency Symbols: Symbols like $, €, £, ¥, and ₹ are replaced with their corresponding words (e.g., "dollar" for $) to maintain textual clarity.

Replacing Digits with Text: Numbers are replaced with their word equivalents (e.g., '1' becomes 'one') for easier analysis.

Removing Non-Alphanumeric Characters: Non-alphanumeric characters (excluding spaces) are removed to keep the text clean.

Counting and Removing Extra Spaces: Extra spaces are detected and removed to ensure the text has consistent spacing.

Removing Stop Words: Common English stop words (e.g., "the", "is", "in") are removed to focus on more meaningful words.
preprocessing\accuracygraphbar.ipynb
This script creates an interactive bar chart using Plotly to visualize the performance metrics of different machine learning models. It compares models like Support Vector Classifier, Logistic Regression, Decision Tree, Naive Bayes, and BERT across four key metrics: Accuracy, Precision, Recall, and F1 Score. The values are displayed as percentages.
Key Points:
Metrics: Accuracy, Precision, Recall, F1 Score.
Models: SVC, Logistic Regression, Decision Tree, Naive Bayes, BERT.
Visualization: A grouped bar chart where each model's performance on the metrics is color-coded for easy comparison.
Customization: The chart includes rotated labels, grouped bars, custom colors, and percentage scores displayed on the y-axis.
# graduation project bert
Project Structure
AbstractModel.py: Abstract base class for models, defining common methods like train, predict, and get_vectorizer.

BertModel.py: Implements a BERT-based model for email classification with options for hyperparameter tuning and data balancing.

DataLoader.py: Class responsible for loading and displaying the dataset information.

DataPreprocessor.py: Handles the preprocessing of data, such as dropping missing values.

ModelTrainer.py: Central class that loads data, trains the model, and evaluates it using k-fold cross-validation.

app.py: Streamlit web application for users to input email content and get predictions (Spam or Ham).
# graduation project 4 algorith
decision_tree_model.py:

Contains the DecisionTreeModel class, which inherits from AbstractModel. It uses either TfidfVectorizer or CountVectorizer to convert text data into features.
Trains a Decision Tree model (DecisionTreeClassifier) and makes predictions. The class also includes methods for setting and retrieving parameters.
logistic_regression_model.py:

Contains the LogisticRegressionModel class, also inheriting from AbstractModel. It uses the same text feature extraction method (TfidfVectorizer or CountVectorizer).
Trains a Logistic Regression model (LogisticRegression) and performs predictions. It has functions to tune and retrieve parameters.
main.py:

This is the main file to run the project. It loads the data, preprocesses the text, and then trains the specified model (Decision Tree, Logistic Regression, etc.) using ModelTrainer.
It can apply SMOTE for data balancing if there's class imbalance.
model_trainer.py:

Contains the ModelTrainer class, which splits the data using StratifiedKFold, trains the model, and evaluates it with 5-fold cross-validation.
Handles text conversion via TfidfVectorizer or CountVectorizer and can also apply SMOTE for balancing the data.
It saves evaluation results (accuracy, precision, recall, F1 score) to a text file and saves the best model.
naive_bayes_model.py:

Contains the NaiveBayesModel class that uses either TfidfVectorizer or CountVectorizer to transform text into features.
Trains a Naive Bayes model (MultinomialNB) and makes predictions.
svc_model.py:

Contains the SvcModel class, which inherits from AbstractModel.
It uses either TfidfVectorizer or CountVectorizer to convert text data into numerical features.
Trains an SVM model (SVC) and makes predictions on the test data.
The class includes methods for setting and retrieving hyperparameters (e.g., kernel type, regularization).
It is flexible in allowing different vectorization types and model parameter adjustments via the set_params and get_params methods.
