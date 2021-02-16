import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import ensemble, preprocessing, metrics

file1 = open('HI-union-train0.k4.tsv', 'r')
Lines = file1.readlines()
pairsList = []
yList = []
strList = []
for l in Lines:
	data = l.split('\t')
	pairs = data[0].split(' ')[0]
	y = data[0].split(' ')[1]
	string = ""
	for d in data[1:]:
		string += (' '.join([d.split()[0]] * int (d.split()[1])))
		string += ' '
	pairsList.append(pairs)
	yList.append(y)
	strList.append(string.strip())


df = pd.DataFrame({'pair':pairsList,'y':yList, 'strList': strList})
print(df)


print("GO GET TFIDF")
vectorizer = TfidfVectorizer(lowercase=False,tokenizer=lambda x: x.split(' '))
X = vectorizer.fit_transform(df['strList'])
print(X.shape)
arr = X.toarray()

X_train, X_test, y_train, y_test = train_test_split(arr, df['y'], test_size=0.33, random_state=42)

forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(X_train, y_train)
test_y_predicted = forest.predict(X_test)
accuracy = metrics.accuracy_score(y_test, test_y_predicted)
print(accuracy)

