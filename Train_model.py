import pandas as pd
import numpy as np
import random
import pickle

import spacy
from spacy.vectors import Vectors

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Loading a tokenizer to have data on each word 
nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "tok2vec"])
tokenizer = nlp.tokenizer
# adding a vector shape for the tokens
empty_vectors = Vectors(shape=(10000, 300))


# Log number to know where we are in the preprocessing
logNum = 0

# Function that will do the preprocessing
# We check if each word is relevent. And if yes we get the vector shape of this word
def preprocessing(text):

    # Print a log values to know where we are
    global logNum
    logNum += 1
    if logNum % 1000 == 0 :
        print("index " + str(logNum))

    doc = nlp(text)

    vectorSum = np.zeros((300))
    for token in doc :
        if (not token.is_stop and len(token) > 2 and token.is_alpha):
            vectorSum = np.add(vectorSum, nlp.vocab[token.lemma].vector)

    return vectorSum


def add_clean(data, list_classes):
    returnData = []

    # Adding a class for non toxic messages
    for values in data[list_classes].values:
        temp = np.array(values)
        if (np.all(temp == 0)):
            returnData.append(np.append(temp, np.array([1])))
        else :
            returnData.append(np.append(temp, np.array([0])))
    
    return returnData


def split_train(X, Y, threshold):
    X_TRAIN = []
    Y_TRAIN = []
    X_TEST = []
    Y_TEST = []

    # Doing the split for train and test
    for i in range (len(X)):
        if random.random() < threshold:
            X_TRAIN.append(X[i])
            Y_TRAIN.append(Y[i])
        else :
            X_TEST.append(X[i])
            Y_TEST.append(Y[i])
    
    return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST


# We load the data into a dataframe
train = pd.read_csv('train.csv')

# Getting the names of all the classes
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Array for the output values (adding 1 output possible for the non toxic)
y_train = add_clean(train, list_classes)

# Doing some preprocessing for the data
print("-- Start preprocessing -- ")
train["text_preprocessed"] = train["comment_text"].apply(preprocessing)

# Having all the data for the gridsearch
all_X = train["text_preprocessed"].tolist()
all_Y = y_train

# Splitting the train and test data
print("Start split of data")
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = split_train(train["text_preprocessed"], y_train, 0.8) # threshold is amount of train

print("length of train : " + str(len(X_TRAIN)))
print("length of test : " + str(len(X_TEST)))

# Creating the ML
model = MLPClassifier()

# Adding parameters to search for the hyper-parameters
parameter_space = {
    'hidden_layer_sizes': [ (50, 100, 50),
                            (100, 100, 100),
                            (300, 100, 50),
                            (300, 50, 6),
                            (100,),
                            (50, 50, 50, 50),
                            (50, 100, 100, 50),
                            (50, 150, 50),
                            (50, 200, 50),
                            (50, 100, 100, 100, 50),
                            (50, 25, 10),
                            (100, 50, 25),
                            (100, 200, 100),
                            (50, 50, 100, 100, 50, 50),
                            (100, 200, 200, 100)],
    'activation': ['tanh'],
    'solver': ['adam'],
    'alpha': [0.05],
    'learning_rate': ['constant'],
}

# # Creating the GirdSearch
# print("Fit gridSearch")
# clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=3, verbose=3, scoring="f1_weighted")
# clf.fit(all_X, all_Y) 

# # Displaying the best parameters and the gridSearch
# print(clf.best_params_)
# print(clf.best_score_)

# # Get the best parameters from the grid search
# best_params = clf.best_params_

# # Create a new MLP model with the best parameters
#  print("Create ML model")
# best_model = MLPClassifier(**best_params)
best_model = MLPClassifier(activation= 'tanh', alpha=0.05, hidden_layer_sizes=(100, 100, 100), learning_rate='constant', solver='adam')

# Train the model
print("Train ML model")
best_model.fit(X_TRAIN, Y_TRAIN)

y_predict = best_model.predict(X_TEST)
print(classification_report(Y_TEST, y_predict))

# Save the model
with open('NLP_BEST_Model.pkl', 'wb') as file:
    pickle.dump(best_model, file)