# hadiel-Al-jadidi-graduation-project-
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
