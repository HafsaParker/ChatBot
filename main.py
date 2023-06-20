# Importing Liberaries
import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Dropout

nltk.download("punkt")
nltk.download("wordnet")

# loading data 
f = open('data.json')
data  = json.load(f)

# processing Data
lm = WordNetLemmatizer()
#lists
ourClass = []
newWords = []
docX = []
docY=[]

for d in data["ourIntents"]:
    for p in d["patterns"]:
        ournewTkns = nltk.word_tokenize(p)
        newWords.extend(ournewTkns)
        docX.append(p)
        docY.append(d["tag"])
        print("ournewTkns",ournewTkns)
        print("newWords",newWords)
        print("docX",docX)
        print("docY",docY)
    if d["tag"] not in ourClass:
        ourClass.append(d["tag"])

newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation]
newWords = sorted(set(newWords))
ourClasses = sorted(set(ourClass))



