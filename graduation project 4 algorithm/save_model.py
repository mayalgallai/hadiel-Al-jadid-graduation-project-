#save_model.py
import pickle
from sklearn.tree import DecisionTreeClassifier

# نفترض أن النموذج هو DecisionTreeClassifier مدرب
model = DecisionTreeClassifier()

# حفظ النموذج باستخدام pickle
with open('best_model_decision_tree.pkl', 'wb') as file:
    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

print("saved.")
