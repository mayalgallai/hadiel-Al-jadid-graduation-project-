from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def get_vectorizer(self):
        pass
