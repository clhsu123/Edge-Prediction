import pandas as pd
import numpy as np
import math
from operator import itemgetter
from collections import defaultdict
from sklearn import preprocessing
from xgboost import XGBClassifier
import time
import argparse
import os
import sys

#Create the parser
my_parser = argparse.ArgumentParser(description = 'Input the folder number and k')
#Add the arguments
my_parser.add_argument('-folder', '--folder', metavar = '{folder}', required = True,  type = str, help = 'The folder number')
my_parser.add_argument('-k','--k', metavar = '{k}', required = True, type = int, help = 'The number of k')
#Execute the parse_arge() method
args = my_parser.parse_args()

#Parse the arguments from commend line and store them in variables
folder = args.folder
N = args.k

#If the arguments are invalid value, show error message and exit the program
if not(folder>='0' and folder<='9' and N>=4 and N<=8):
    print('folder number should be between 0 and 9, N should be between 4 and 8 inclusively. Exit.')
    exit(0)

#read orbit map from k = 4 to k = 8 and store each map in orbitmap array
orbitMap = []
for i in range(4, N+1):
    file_name = "./orbitmap/orbit_map"+str(i)+".txt"
    orbitMap.append(np.loadtxt(file_name, skiprows = 1, dtype = int))
#contruct string named file_number from 4 to k.  E.g. k = 6, file_number = "456"
file_number = ""
for i in range(4, N+1):
	file_number+=str(i)

#Store the name of test files and train files from specific folder
test_filename = "./F"+folder+"/HI-union-test"+folder+".el"
train_filename = "./F"+folder+"/HI-union-train"+folder+".k"+file_number+".tsv"
Topk = 500

#Convert motif IDs to orbit IDs. E.g. [4:5:6:7 4] -> [k4_5 4, k4_6 4]
def convertMotif(motifList):
	#Initialze a dictionary and set the default value to 0
	k_to_cnt = {}
	k_to_cnt = defaultdict(lambda: 0,  k_to_cnt)
	#parse each motif ID. E.g. [4:5:6:7 4]
	for m in motifList:
		k, row, col1, col2 = map(int, m.split(" ")[0].split(":"))
		count = int (m.split(" ")[1])
		#Map the row and col to orbit IDs in orbit map
		k1, k2 = "k"+str(k)+"_"+ str(orbitMap[k-4][row][col1]), "k"+str(k)+"_"+str(orbitMap[k-4][row][col2])
		#Count the frequency of each orbit ID using hash table
		k_to_cnt[k1] += count
		k_to_cnt[k2] += count
	#Return a list of orbit ID and its frequency. E.g. [k4_3 5, k4_4 2, k5_3 7]
	return [ k+" "+str(v) for k, v in k_to_cnt.items()]

#Compute the IDF(For the definition of IDF, refer to the document) for each orbit ID, store them in hash table and return it
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
#Parse the orbit ID list and return the TFIDF (For the definition fo TFIDF, refer to the document)array ready to be traind
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
	#Parse each line of input datas
	for l in Lines:
		data = l.rstrip().split('\t')
		pairs = data[0].split(' ')[0]
		y = data[0].split(' ')[1]
		motifs = convertMotif(data[1:])
		pairsList.append(pairs)
		yList.append(y)
		motifList.append(motifs)
	return pairsList, yList, motifList

#Read the file and store them in variables (as arrays)
print("READ FILE")
pairsList, yList, motifList = readFile()
start_time = time.time()

#Compute IDF Dictionary and store necessary informations
print("COMPUTE IDF")
FullDict = computeIDF(motifList)
numdict = {4:3, 5:5, 6:7, 7:40, 8:600}
num  = numdict[N]
idfDict = dict(sorted(FullDict.items(), reverse = True, key = itemgetter(1))[:num])

#Declare a 2D array and initialize it to 0 with the length of the first dimension being number of input data pieces and the length of the second dimension being the length of IDF Dictionary
TFIDF_arr = np.zeros((len(yList), len(idfDict)))
uniMotifList = list(idfDict.keys())
uniMotifIdx = dict(zip(uniMotifList, range(0, len(uniMotifList))))

print("COMPUTE TFIDF")
TFIDF_arr = computeTFIDF(TFIDF_arr, uniMotifList, uniMotifIdx, idfDict, motifList)
TFIDF_arr = preprocessing.normalize(TFIDF_arr, norm="l1")
yList = list(map(int, yList)) 
#train
print("Start Train")
model = XGBClassifier(eta = 0.1, verbosity = 0, use_label_encoder =False)
model.fit(TFIDF_arr, yList)

#evaluate
print("Start Evaluate")
#Parse the yList and store the index of elements with value 0 in array testIDx
testIdx = [i for i, x in enumerate(yList) if x == 0]
#Use the testIDx to retrieve the values of TFIDF array and store them in array testArr
testArr = np.take(TFIDF_arr, testIdx, axis = 0)
#Use the testIDx to retrieve the values of pairsList array and store them in array testPairs
testPairs = [pairsList[i] for i in testIdx]

#Generate predictions for testArr, these are the predictions for data piece whose y are 0
test_y_predicted = model.predict_proba(testArr)
#Store the index of the top k entry values sorted by the predicted values (here the predicted values mean the confidence of having edges)
ind = np.argpartition(test_y_predicted[:,1], -Topk)[-Topk:]
#Store the list of predicted edges
output = [testPairs[i] for i in ind]
# match
testData = pd.read_table(test_filename, sep = ' ', header = None)
ans = list(testData[0].apply(lambda x: x.split('\t')[1] + ':'+ x.split('\t')[0]))
print("Folder = "+folder+", k = "+file_number+":")
print(len(list(set(ans) & set(output))))
print("--- %s seconds ---" % (time.time() - start_time))
