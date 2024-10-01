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

