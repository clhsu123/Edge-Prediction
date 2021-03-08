import pandas as pd
import numpy as np
import math
from operator import itemgetter
from collections import defaultdict
from sklearn import preprocessing
import sys
from xgboost import XGBClassifier
import time

if not len(sys.argv) == 3:
	print("Wrong number of argument passed to commend line, please follow the format:")
	print("python TFIDF.py folder_number file_number")
	sys.exit()

folder = str(sys.argv[1])
N = int(sys.argv[2])

orbitMap = []
for i in range(4, N+1):
    file_name = "./orbitmap/orbit_map"+str(i)+".txt"
    orbitMap.append(np.loadtxt(file_name, skiprows = 1, dtype = int))
file_number = ""
for i in range(4, N+1):
	file_number+=str(i)

test_filename = "./F"+folder+"/HI-union-test"+folder+".el"
train_filename = "./F"+folder+"/HI-union-train"+folder+".k"+file_number+".tsv"
Topk = 500

def convertMotif(motifList):
	k_to_cnt = {}
	k_to_cnt = defaultdict(lambda: 0,  k_to_cnt)
	for m in motifList:
		k, row, col1, col2 = map(int, m.split(" ")[0].split(":"))
		count = int (m.split(" ")[1])
		k1, k2 = "k4_"+ str(orbitMap[k-4][row][col1]), "k4_"+str(orbitMap[k-4][row][col2])
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
			if m not in uniMotifIdx.keys(): continue; 
			count = data.split()[1]
			TFIDF_arr[idx][uniMotifIdx[m]] = (idfDict[m]) * int(count)
	return TFIDF_arr

def readFile():
	trainfile = open(train_filename, 'r')
	Lines = trainfile.readlines()
	pairsList = []
	yList = []
	motifList =[]
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
start_time = time.time()
idfDict = computeIDF(motifList)
num  = int(len(idfDict)/2)
idfDict = dict(sorted(idfDict.items(), reverse = True, key = itemgetter(1))[:num])

TFIDF_arr = np.zeros((len(yList), len(idfDict)))
uniMotifList = list(idfDict.keys())
uniMotifIdx = dict(zip(uniMotifList, range(0, len(uniMotifList))))

TFIDF_arr = computeTFIDF(TFIDF_arr, uniMotifList, uniMotifIdx, idfDict, motifList)
TFIDF_arr = preprocessing.normalize(TFIDF_arr, norm="l1")
yList = list(map(int, yList)) 
#train
print("Start Train")
model = XGBClassifier(eta = 0.1, verbosity = 0, use_label_encoder =False)
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
testData = pd.read_table(test_filename, sep = ' ', header = None)
ans = list(testData[0].apply(lambda x: x.split('\t')[1] + ':'+ x.split('\t')[0]))
print("Folder = "+folder+", k = "+file_number+":")
print(len(list(set(ans) & set(output))))
print("--- %s seconds ---" % (time.time() - start_time))