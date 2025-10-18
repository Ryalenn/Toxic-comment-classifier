import pickle
import numpy as np

import spacy
from spacy.vectors import Vectors


# Loading a tokenizer to have data on each word 
nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "tok2vec"])
tokenizer = nlp.tokenizer
# adding a vector shape for the tokens
empty_vectors = Vectors(shape=(10000, 300))

# Function that will do the preprocessing
# We check if each word is relevent. And if yes we get the vector shape of this word
def preprocessing(text):

    doc = nlp(text)

    vectorSum = np.zeros((300))
    for token in doc :
        if (not token.is_stop and len(token) > 2 and token.is_alpha):
            vectorSum = np.add(vectorSum, nlp.vocab[token.lemma].vector)

    return vectorSum


# Load the model back using pickle
with open('NLP_BEST_Model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# The sencence that's gonna be analysed
sentence = "jew"
preprocessedSentence = preprocessing(sentence)

# Perform predictions on the test set
y_pred = loaded_model.predict([preprocessedSentence])

# Getting the names of all the classes
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "not toxic"]

# Print the predictions
print("Your message has been detected with this classes : ")
for i in range (len(list_classes)):
    if (y_pred[0][i] == 1):
        print(list_classes[i])