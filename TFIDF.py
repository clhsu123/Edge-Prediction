import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from sklearn import preprocessing
Topk = 500
orbitMap4 = np.loadtxt("./orbit_map4.txt", skiprows = 1, dtype=int)

def convertMotif(motifList):
	k_to_cnt = {}
	k_to_cnt = defaultdict(lambda: 0,  k_to_cnt)
	for m in motifList:
		k, row, col1, col2 = map(int, m.split(" ")[0].split(":"))
		count = int (m.split(" ")[1])
		k1, k2 = "k4_"+ str(orbitMap4[row-2][col1]), "k4_"+str(orbitMap4[row-2][col2])
		k_to_cnt[k1] += count
		k_to_cnt[k2] += count
	return [ k+" "+str(v) for k, v in k_to_cnt.items()]


def computeIDF(documents):
    N = len(documents)
    idfDict = {}
    for document in documents:
        for motif in document:
            key = motif.split()[0]
            idfDict.setdefault(key, 0)
            idfDict[key] +=1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(TFIDF_arr, uniMotifList, uniMotifIdx, idfDict, motifList):
	for idx, motifs in enumerate(motifList):
		for data in motifs:
			m = data.split()[0]
			count = data.split()[1]
			TFIDF_arr[idx][uniMotifIdx[m]] = int(count)
	return TFIDF_arr

def norml2(X):
	l2norm = np.sqrt((X * X).sum(axis=1))
	X /= l2norm.reshape(len(X),1)
	return X

def readFile():
	file1 = open('HI-union-train0.k4.tsv', 'r')
	Lines = file1.readlines()
	pairsList = []
	yList = []
	motifList = []
	for l in Lines:
		data = l.rstrip().split('\t')
		pairs = data[0].split(' ')[0]
		y = data[0].split(' ')[1]
		motifs = convertMotif(data[1:])
		pairsList.append(pairs)
		yList.append(y)
		motifList.append(motifs)
	return pairsList, yList, motifList

pairsList, yList, motifList = readFile()
idfDict = computeIDF(motifList)
TFIDF_arr = np.zeros((len(yList), len(idfDict)))
uniMotifList = list(idfDict.keys())
uniMotifIdx = dict(zip(uniMotifList, range(0, len(uniMotifList))))

TFIDF_arr = computeTFIDF(TFIDF_arr, uniMotifList, uniMotifIdx, idfDict, motifList)
TFIDF_arr = preprocessing.normalize(TFIDF_arr, norm="l1")
yList = list(map(int, yList)) 
#train
print("Start Train")
model = LogisticRegression(max_iter=1000)
model.fit(TFIDF_arr, yList)

#evaluate
print("Start Evaluate")
testIdx = [i for i, x in enumerate(yList) if x == 0]
testArr = np.take(TFIDF_arr, testIdx, axis = 0)
testPairs = [pairsList[i] for i in testIdx]

test_y_predicted = model.predict_proba(testArr)
ind = np.argpartition(test_y_predicted[:,1], -Topk)[-Topk:]
output = [testPairs[i] for i in ind]
# match
testData = pd.read_table("./HI-union-test0.el", sep = ' ', header = None)
ans = list(testData[0].apply(lambda x: x.split('\t')[1] + ':'+ x.split('\t')[0]))
print(list(set(ans) & set(output)))
