import importlib
import itertools
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np

class ModelTrainer:
    def __init__(self, X, y, model_name, use_hyperparameters=False, balance_data=False):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.use_hyperparameters = use_hyperparameters
        self.balance_data = balance_data  # إضافة خيار موازنة البيانات
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.accuracies = []
        self.f1_scores = []
        self.best_f1_score = 0
        self.best_model = None
        self.best_fold = 0

    def get_model_class(self):
        module = importlib.import_module(f'{self.model_name}_model')
        class_name = ''.join([part.capitalize() for part in self.model_name.split('_')])
        model_class = getattr(module, f'{class_name}Model')
        return model_class

    def train_and_evaluate(self):
        model_class = self.get_model_class()
        fold = 1
        for train_index, test_index in self.skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            print(f"Fold {fold}: Training data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")

            # تمرير الهايبر براميترز للنموذج إذا كان الخيار مفعلًا
            model = model_class(use_hyperparameters=self.use_hyperparameters, balance_data=self.balance_data)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            self.accuracies.append(accuracy)

            f1 = f1_score(y_test, y_pred, average='weighted')
            self.f1_scores.append(f1)

            if f1 > self.best_f1_score:
                self.best_f1_score = f1
                self.best_model = model
                self.best_fold = fold

            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)

            with open('classification_report.txt', 'a', encoding='utf-8') as report_file:
                report_file.write(f"Fold {fold}:\n")
                report_file.write(f"Accuracy: {accuracy:.4f}\n")
                report_file.write(f"F1-Score: {f1:.4f}\n")
                report_file.write("Confusion Matrix:\n")
                report_file.write(f"{cm}\n")
                report_file.write("Classification Report:\n")
                report_file.write(f"{cr}\n")
                report_file.write("\n" + "="*50 + "\n\n")

            # رسم المصفوفة (Confusion Matrix)
            plt.figure(figsize=(8, 6))
            classes = np.unique(self.y)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - Fold {fold}")
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_fold_{fold}.png')
            plt.close()

            fold += 1

        average_accuracy = sum(self.accuracies) / len(self.accuracies)
        average_f1_score = sum(self.f1_scores) / len(self.f1_scores)

        with open('classification_report.txt', 'a', encoding='utf-8') as report_file:
            report_file.write("K-Folds:\n")
            for i, (accuracy, f1) in enumerate(zip(self.accuracies, self.f1_scores), 1):
                report_file.write(f"Fold {i}: Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}\n")

            report_file.write(f"\naverage_accuracy: {average_accuracy:.4f}\n")
            report_file.write(f"average_f1_score: {average_f1_score:.4f}\n")
            report_file.write(f"\nbest_fold {self.best_fold} with F1-Score: {self.best_f1_score:.4f}\n")

        # رسم f1-score عبر جميع الطيات
        plt.figure(figsize=(8, 6))
        plt.imshow(np.array([self.f1_scores]), cmap=plt.cm.Blues, interpolation='nearest')
        plt.title(f"F1-Score across Folds")
        plt.colorbar()
        plt.xlabel('Fold')
        plt.ylabel('F1-Score')
        plt.tight_layout()
        plt.savefig('f1_score_across_folds.png')
        plt.close()

        joblib.dump(self.best_model, f'best_model_{self.model_name}.pkl')
        joblib.dump(self.best_model.get_vectorizer(), f'best_vectorizer_{self.model_name}.pkl')
