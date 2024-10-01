class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        self.df = self.df.dropna(subset=['Masseges', 'Category'])  # Adjust column names as per your dataset
        X = self.df['Masseges']  # Adjust column name
        y = self.df['Category']  # Adjust column name
        print("\nMissing values in X:", X.isnull().sum())
        print("Missing values in y:", y.isnull().sum())
        return X, y
