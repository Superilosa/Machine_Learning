from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

A = "I love horror movies"
B = "Lights out is a horror movie"

a = word_tokenize(A)
b = word_tokenize(B)
sw = stopwords.words('english')
a = {w for w in a if not w in sw}
b = {w for w in b if not w in sw}
print(a,b)
s = a.union(b)
print(s)
la=lb=[]
for w in s:
    if w in a:
        la.append(1)
    else:
        la.append(0)
    if w in b:
        lb.append(1)
    else:
        lb.append(0)
c=0
for i in range(len(s)):
    c += la[i]*lb[i]
cos = c / float(((sum(la))*sum(lb))**0.5)
print("similarity: ",cos)