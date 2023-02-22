import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle

def extractTestData(ixD, pqSets, tests, itr):
	newMatrix = np.dot(pqSets[0],pqSets[1])

	ixs = ixD[itr[0]]
	nm = newMatrix[ixs[0][0]:ixs[0][1], ixs[1][0]:ixs[1][1]]
	test = tests[0][ixs[0][0]:ixs[0][1], ixs[1][0]:ixs[1][1]]

	columns = ["Actual", "Predicted"]

	recommend = []
	for i in range(len(nm)):
		rec = pd.DataFrame(columns=columns)
		for j in range(len(nm[0])):
			if test[i][j] > 0:
				rec = rec.append({"Actual":test[i][j], "Predicted":float(str(round(nm[i][j], 2)))}, ignore_index = True)

		recommend.append(rec)


	newMatrix = np.dot(pqSets[2],pqSets[3])

	ixs = ixD[itr[1]] 
	nm = newMatrix[ixs[0][0]:ixs[0][1], ixs[1][0]:ixs[1][1]]
	test = tests[1][ixs[0][0]:ixs[0][1], ixs[1][0]:ixs[1][1]]


	for i in range(len(nm)):
		rec = pd.DataFrame(columns=columns)
		for j in range(len(nm[0])):
			if test[i][j] > 0:
				rec = rec.append({"Actual":test[i][j], "Predicted":float(str(round(nm[i][j], 2)))}, ignore_index = True)

		pd.concat([recommend[i],rec])

	return recommend

def calculateMetrics(recommend):
	totNum = len(recommend)
	precision = 0
	recall = 0
	for rec in recommend:
		rec = rec.sort_values("Predicted", ascending = False)
		rec = rec.loc[rec["Predicted"] <= 5]

		pred = rec.loc[rec["Predicted"] >= 3.5]
		totRelNum = len(pred.loc[pred["Actual"] >= 3.5])
		pred = pred.head(10)
		rec = rec.head(10)
		relNum = len(pred.loc[pred["Actual"] >= 3.5])
		total = len(rec)

		##Calculate Precision##
		if total > 0:
			prec = float(relNum)/float(total)
			precision += prec

		##Calculate Recall##
		if totRelNum > 0:
			rec = float(relNum)/float(totRelNum)
			recall += rec

	precision /= totNum
	recall /= totNum

	return precision, recall

##MAIN##

##Import Data##
ratings = pd.read_csv('ratings.csv', quotechar='"', sep=',')

##List of Movie IDs##
mID = sorted(ratings['movieId'].unique())
mNum = len(mID)
##List of User IDs##
uID = sorted(ratings['userId'].unique())
uNum = len(uID)
##Number of Features##
fNum = 75
a = 0.01
b = 0.02
steps = 750

print("Creating Matrix...")
##Create Ratings Matrix (Rows: UserID, Columns: MoveID, Content: Ratings)##
mainMatrix = []
for user in uID:
	uVector = [0] * len(mID)
	for i, r in ratings.loc[ratings['userId'] == user].iterrows():
		uVector[mID.index(r['movieId'])] += r['rating']
	mainMatrix.append(uVector)

mainMatrix = np.array(mainMatrix)

#Pickle Matrix##
pickle.dump(mainMatrix, open('mainMatrix.sav','wb'))

##Load Matrix##
mainMatrix = pickle.load(open('mainMatrix.sav', 'rb'))

##Spliting Data Into Training and Test Sets##
row = uNum
col = mNum
mRow = int(row/2)
mCol = int(col/2)

ixD = {
	0: [[0,mRow],[0,mCol]],
	1: [[0,mRow],[mCol,col]],
	2: [[mRow,row],[0,mCol]],
	3: [[mRow,row],[mRow,col]]
}

trainSet = []
testSet = []

for fold in range(4):
	ixs = ixD[fold]

	trainMask = np.full((row,col),1)
	trainMask[ixs[0][0]:ixs[0][1], ixs[1][0]:ixs[1][1]] = 0
	testMask = 1 - trainMask

	x_train = mainMatrix.copy()
	x_train[trainMask == 0] = 0
	trainSet.append(x_train)

	x_test = mainMatrix.copy()
	x_test[testMask == 0] = 0
	testSet.append(x_test)

#Pickle Train/Test##
pickle.dump(trainSet, open('trainSet.sav','wb'))
pickle.dump(testSet, open('testSet.sav','wb'))

##Load Train##
trainSet = pickle.load(open('trainSet.sav', 'rb'))	

#Training Model##
pSet = []
qSet = []
print("Training Model...")
for fold in range(4):
	##User Feature Matrix##
	P = np.random.rand(uNum,fNum)
	##Movie Feature Matrix##
	Q = np.random.rand(mNum,fNum)
	Q = Q.T
	for step in range(steps):
		print("Step: " + str(step))
		for i in range(len(trainSet[fold])):
			for j in range(len(trainSet[fold][i])):
				if(trainSet[fold][i][j] > 0):
					err = trainSet[fold][i][j] - np.dot(P[i,:],Q[:,j])

					for k in range(fNum):
						pTemp = P[i][k] + a * (2 * err * Q[k][j] - b * P[i][k])
						Q[k][j] = Q[k][j] + a * (2 * err * P[i][k] - b * Q[k][j])
						P[i][k] = pTemp

	pSet.append(P)
	qSet.append(Q)

pickle.dump(pSet, open('pSet.sav','wb'))
pickle.dump(qSet, open('qSet.sav','wb'))

##Testing Model##
print("Testing...")
pSet = pickle.load(open('pSet.sav', 'rb'))
qSet = pickle.load(open('qSet.sav', 'rb'))
testSet = pickle.load(open('testSet.sav', 'rb'))

itr1 = [0,1]
itr2 = [2,3]
testFirst = [testSet[0], testSet[1]]
testSecond = [testSet[2], testSet[3]]
pqFirst = [pSet[0], qSet[0], pSet[1], qSet[1]]
pqSecond = [pSet[2], qSet[2], pSet[3], qSet[3]]
recommend1 = extractTestData(ixD, pqFirst, testFirst, itr1)

precision1, recall1 = calculateMetrics(recommend1)

print("Precision1: " + str(precision1))
print("Recall1: " + str(recall1))

recommend2 = extractTestData(ixD, pqSecond, testSecond, itr2)
precision2, recall2 = calculateMetrics(recommend2)

print("Precision2: " + str(precision2))
print("Recall2: " + str(recall2))


precision = (precision1 + precision2) / 2
recall = (recall1 + recall2) / 2

print("Precision: " + str(precision))
print("Recall: " + str(recall))