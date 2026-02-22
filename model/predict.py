import pickle
import pandas as pd


with open('model/model.pkl', 'rb') as f:
    model= pickle.load(f)

class_labels= model.classes_.tolist()

MODEL_VERSION = "1.0.0"


def predict_output(user_input:dict):
    df= pd.DataFrame([user_input])
    
    predicted_class = model.predict(df)[0]

    #get probabilities for all classes
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)

    #create mapping of class labels to probabilities
    class_probs= dict(zip(model.classes_, map(lambda x: round(x, 4), probabilities)))
    return {
        'predicted_category': predicted_class,
        'confidence': round(confidence, 4),
        'class_probabilities': class_probs
    }