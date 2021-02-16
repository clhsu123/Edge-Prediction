import pandas as pd
import numpy as np
import math

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
			TFIDF_arr[idx][uniMotifIdx[m]] = (idfDict[m]+1) * int(count)
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
		motifs = data[1:]
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
TFIDF_arr = norml2(TFIDF_arr)



