#may moktar algallai 
#hadil ali aljadid
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from bert_model import BertModel
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

if __name__ == "__main__":
    file_path = 'spammail.csv'
    model_name = 'bert'  
    use_hyperparameters = False  # تغيير هذه القيمة إذا كنت لا تريد استخدام الهايبر براميترز
    balance_data = False  # تغيير هذه القيمة لتفعيل أو تعطيل موازنة البيانات

    # تحميل البيانات
    data_loader = DataLoader(file_path)
    data_loader.load_data()

    # معالجة البيانات
    data_preprocessor = DataPreprocessor(data_loader.df)
    X, y = data_preprocessor.preprocess()

    # تدريب النموذج وتقييمه
    model_trainer = ModelTrainer(X, y, model_name, use_hyperparameters, balance_data)
    model_trainer.train_and_evaluate()

