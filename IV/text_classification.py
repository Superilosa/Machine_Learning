import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB

def text_process(s):
    p = [c for c in s if c not in string.punctuation]
    p = ''.join(p)
    return [w for w in p.split() if w.lower() not in stopwords.words('english')]

data = pd.read_csv("https://raw.githubusercontent.com/bigmlcom/python/master/data/spam.csv",
                   encoding='UTF-8',skiprows=1,sep='\t',names=['label','message'],engine='python')
# print(data.groupby('label').describe())
data['message_token'] = data['message'].apply(text_process)
wcs = CountVectorizer(analyzer=text_process).fit_transform(data['message']).toarray()
# wcs აღნიშნავს რომელ წინადადებაში რომელი სიტყვა რამდენჯერ გვხვდება
# მისი ზომა გამოდის (წინადადებების რაოდენობა X უნიკალურ სიტყვების რაოდენობაზე). დამტკიცება:
# print(data.shape)
# uniq = []
# for i in range(data.shape[0]):
#     for w in data['message_token'][i]:
#         if w not in uniq:
#             uniq.append(w)
# print(len(uniq))
# print(wcs.shape)
message_matrix = TfidfTransformer().fit_transform(wcs)
le = LabelEncoder()
data['label']=le.fit_transform(data['label'])
X_train, X_test, y_train, y_test = train_test_split(message_matrix,data['label'],test_size=0.3,random_state=2)
model = MultinomialNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))