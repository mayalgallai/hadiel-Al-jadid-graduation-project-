from collections import Counter
from sklearn.utils import resample
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
from abstract_model import AbstractModel

# Define compute_metrics function outside the class
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    precision = precision_score(p.label_ids, preds, average='weighted')
    recall = recall_score(p.label_ids, preds, average='weighted')
    f1 = f1_score(p.label_ids, preds, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts.iloc[item]
        label = self.labels.iloc[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertModel(AbstractModel):
    def __init__(self, use_hyperparameters=False, balance_data=False):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.max_len = 128
        self.batch_size = 16
        self.epochs = 3
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.label_encoder = LabelEncoder()
        self.use_hyperparameters = use_hyperparameters
        self.balance_data = balance_data  # إضافة خيار موازنة البيانات

    def balance_dataset(self, X, y):
        # حساب توزيع الفئات
        class_distribution = Counter(y)
        print(f"Original class distribution: {class_distribution}")

        # دمج البيانات والنصوص معًا لإعادة التوزيع
        df = pd.DataFrame({"text": X, "label": y})

        # الحصول على الأقل تمثيلاً والفئة الأكبر
        minority_class = df['label'].value_counts().idxmin()
        majority_class = df['label'].value_counts().idxmax()

        # فصل البيانات حسب الفئات
        df_minority = df[df['label'] == minority_class]
        df_majority = df[df['label'] == majority_class]

        # زيادة عينات الفئة الأقل (Oversampling)
        df_minority_upsampled = resample(df_minority, 
                                         replace=True,    # عينات مكررة
                                         n_samples=len(df_majority), # نفس عدد عينات الفئة الأكبر
                                         random_state=42)

        # دمج العينات
        df_balanced = pd.concat([df_majority, df_minority_upsampled])

        print(f"Balanced class distribution: {Counter(df_balanced['label'])}")

        # إعادة توزيع X و y بعد الموازنة
        return df_balanced['text'], df_balanced['label']

    def train(self, X_train, y_train):
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # موازنة البيانات إذا كان الخيار مفعلًا
        if self.balance_data:
            X_train, y_train_encoded = self.balance_dataset(X_train, y_train_encoded)

        train_texts, val_texts, train_labels, val_labels = train_test_split(X_train, y_train_encoded, test_size=0.1)

        train_dataset = BertDataset(train_texts, pd.Series(train_labels), self.tokenizer, self.max_len)
        val_dataset = BertDataset(val_texts, pd.Series(val_labels), self.tokenizer, self.max_len)

        if not self.use_hyperparameters:
            # Training without hyperparameters
            training_args = TrainingArguments(
                output_dir='./results',
                learning_rate=2e-5,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.epochs,
                weight_decay=0.01,
                logging_dir='./logs',
                evaluation_strategy="epoch",
                warmup_steps=500,
                adam_epsilon=1e-8,
                max_grad_norm=1.0
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics
            )

            trainer.train()

            eval_result = trainer.evaluate(eval_dataset=val_dataset)
            accuracy = eval_result.get("eval_accuracy", None)
            if accuracy is None:
                print("Accuracy metric not found. Available keys:", eval_result.keys())
            else:
                print(f"Accuracy: {accuracy}")
        
        else:
            # Manual random search for hyperparameters
            param_distributions = {
                'learning_rate': [1e-5, 2e-5, 3e-5],
                'num_train_epochs': [3, 4, 5],
                'per_device_train_batch_size': [16, 32],
                'warmup_steps': [100, 500, 1000],
                'weight_decay': [0.01, 0.05, 0.1]
            }

            best_accuracy = 0
            best_params = None

            for _ in range(1):  # Try 10 different hyperparameter combinations
                current_params = {key: random.choice(values) for key, values in param_distributions.items()}
                
                training_args = TrainingArguments(
                    output_dir='./results',
                    learning_rate=current_params['learning_rate'],
                    per_device_train_batch_size=current_params['per_device_train_batch_size'],
                    per_device_eval_batch_size=current_params['per_device_train_batch_size'],
                    num_train_epochs=current_params['num_train_epochs'],
                    weight_decay=current_params['weight_decay'],
                    logging_dir='./logs',
                    evaluation_strategy="epoch",
                    warmup_steps=current_params['warmup_steps'],
                    adam_epsilon=1e-8,
                    max_grad_norm=1.0
                )

                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=compute_metrics
                )

                trainer.train()

                eval_result = trainer.evaluate(eval_dataset=val_dataset)
                accuracy = eval_result.get("eval_accuracy", None)
                
                if accuracy is None:
                    print("Accuracy metric not found. Available keys:", eval_result.keys())
                else:
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = current_params
                print(f"Best hyperparameters: {best_params}")
                print(f"Best accuracy: {best_accuracy}")

    def predict(self, X_test):
        test_dataset = BertDataset(X_test, pd.Series([0]*len(X_test)), self.tokenizer, self.max_len)
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(test_dataset)
        
        predictions_tensor = torch.tensor(predictions.predictions)
        y_pred = torch.argmax(predictions_tensor, axis=1)
        
        y_pred_labels = self.label_encoder.inverse_transform(y_pred.numpy())
        return y_pred_labels

    def get_vectorizer(self):
        return self.tokenizer
