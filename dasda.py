import numpy as np
import math
import sys
import queue
from lxml.html import document_fromstring

#feature rule:(index, split), ex: (x5, 0.5)==>(4,0.5)

#each node of a decision tree has an array of size 23
class node():

	def __init__(self,vecs):
		self.vecs = vecs
		self.left = None
		self.right = None
		self.rule = None
		self.bound = None
		self.isLeaf = False
		#print(len(self.vecs[542]))
		#calculate maxium labels in this node
		numLabs = self.labels()
		if len(numLabs) == 1:
			self.isLeaf = True

		num1 = 0
		num0 = 0
		for item in self.vecs:
			if item[22] == 1:
				num1 += 1
			else:
				num0 += 1
		self.maxLabel = (1 if num1 >= num0 else 0)
		self.findRule()

	def findRule(self):
		labs = []
		axis = []

		i = 0;
		while(i<len(self.vecs[0])-1):
			v = []
			# print(i,"-->")
			for item in self.vecs:
				v.append(item[i])
			axis.append(v)
			# print(len(v))
			# print('\n')
			i += 1
		
		for item in self.vecs:
			labs.append(item[len(self.vecs[0])-1])

		labs = list(set(labs))
		# print(labs)

		minE = sys.maxsize

		feature = []
		for item in axis:
			# print("-->",len(item))
			item = list(set(item))
			item.sort()
			# print(len(item))
			b = []
			for i in range(0,len(item)-1):
				mid = (item[i] + item[i+1]) * 0.5
				b.append(mid)
			feature.append(b)
		print(len(feature[0]))

		for i in range(0,len(feature)):
			print(i+1)
			for j in range(0, len(feature[i])):
				entropy = self.entropy(i, feature[i][j], labs)
				if entropy < minE:
					minE = entropy
					self.rule = i;
					self.bound = feature[i][j]

		return [self.rule, self.bound]

			
	def entropy(self, i, mid, labs):
		numLess = 0.0
		numGreat = 0.0
		for item in self.vecs:
			if item[i] < mid:
				numLess += 1
			else:
				numGreat += 1

		# print(numLess/len(self.vecs),numGreat/len(self.vecs),len(self.vecs))
		
		#P(Y | Xi < mid)
		prob_y_less = []
		for l in labs:
			condProb = 0.0
			for item in self.vecs:
				if item[i] < mid and item[len(item)-1] == l:
					condProb += 1
					# print(condProb)
			condProb = condProb / numLess
			# print(condProb)
			prob_y_less.append(condProb)
		# print(len(prob_y_less))


		#P(Y | xi >= mid)
		prob_y_great = []
		for l in labs:
			condProb = 0.0
			for item in self.vecs:
				if item[i] >= mid and item[len(item)-1] == l:
					condProb += 1

			condProb = condProb / numGreat
			prob_y_great.append(condProb)

		# print(prob_y_less)

		# H(Y | Xi < mid)
		H_less = 0.00
		for item in prob_y_less:
			if item == 0:
				H_less -= 0
			else:
				H_less -= item * math.log(item, 2.0)

		# #H(Y | (xi >= mid))
		H_great = 0.00
		for item in prob_y_great:
			if item == 0:
				H_great -= 0
			else:
				H_great -= item * math.log(item, 2.0)

		#H(Y | (xi, mid))
		left = numLess*1.00 / len(self.vecs)*1.00
		right = numGreat / len(self.vecs)
		return (left*H_less + right*H_great)

	
	def labels(self):
		labs = []
		for item in self.vecs:
			labs.append(item[22])
		labs = list(set(labs))
		return labs


class DT():

	def __init__(self,root):
		self.root = root
		self.build(self.root)

	def build(self,n):
		print(n.rule, n.bound)
		if(n.isLeaf is True):
			return
		else:
			i = n.rule
			bound = n.bound

			left = []
			right = []
			for item in n.vecs:
				if item[i] < bound:
					left.append(item)
				else:
					right.append(item)

			lsub = node(left)
			rsub = node(right) 

			n.left = lsub
			n.right = rsub

			self.build(n.left)
			self.build(n.right)

	def predict(self, vec):
		temp = self.root
		while(temp.isLeaf is False):
			i = temp.rule
			bound = temp.bound
			# print(vec)
			# print(i,bound)
			if vec[i] < bound:
				temp = temp.left
			else:
				temp = temp.right
		# print(temp.labels()[0])
		return temp.maxLabel

	def prune(self, replaceNode,dataSet,verr, testDataSet):
		replaceNode.isLeaf = True
		total = len(dataSet)
		err = 0
		for item in dataSet:
			# print(index," evaluate error")
			res = self.predict(item)
			if res != item[len(item)-1]:
				err += 1
		newVerr = err / len(dataSet)
		print("validation error ", newVerr)

		err = 0
		for item in testDataSet:
			res = self.predict(item)
			if res != item[len(item)-1]:
				err += 1
		terr = err / len(testDataSet)
		print("test error: ", terr)

		if newVerr > verr:
			replaceNode.isLeaf = False
		else:
			verr = newVerr
		return verr


		



#####################	with pruning   ########################
def pruning(dt):
	dataSet = []
	f = open('hw3validation.txt', 'r')
	for line in f.readlines():
		line = line.strip('\n')

		floats = line.split(' ')

		vecs = [float(i) for i in floats[0:23]]

		dataSet.append(vecs)
	f.close()

	# calculate violdation error:
	total = len(dataSet)
	err = 0
	index = 1
	for item in dataSet:
		# print(index," evaluate error")
		res = dt.predict(item)
		if res != item[len(item)-1]:
			err += 1
		index += 1

	print("intial validation error: ",err / len(dataSet))
	verr = err / len(dataSet)

	test = open('hw3test.txt', 'r')
	testDataSet = []
	for line in test.readlines():
		line = line.strip('\n')
		floats = line.split(' ')

		testVec = [float(i) for i in floats[0:23]]
		testDataSet.append(testVec)
	test.close()

	trie = dt
	q = queue.Queue()
	x = trie.root.left
	if x is not None:
		q.put(x)
	x = trie.root.right
	if x is not None:
		q.put(trie.root.right)
	while(not q.empty()):
		x = q.get()
		verr = trie.prune(x,dataSet,verr, testDataSet)
		if x.isLeaf is True:
			continue
		if x.left is not None:
			q.put(x.left)
		if x.right is not None:
			q.put(x.right)






############################################################
#####       Without Pruning		######################
def readData():
	dataSet = []
	f = open('hw3train.txt', 'r')
	for line in f.readlines():
		line = line.strip('\n')

		floats = line.split(' ')

		vecs = [float(i) for i in floats[0:23]]

		dataSet.append(vecs)


	f.close()
	#print(len(dataSet))

	#print(list(set(labs)))
	root = node(dataSet)
	dt = DT(root)
	# pasing the tree to pruning function
	pruning(dt)
	# total = len(dataSet)
	# err = 0
	# index = 1
	# for item in dataSet:
	# 	# print(index," evaluate error")
	# 	res = dt.predict(item)
	# 	if res != item[len(item)-1]:
	# 		err += 1
	# 	index += 1

	# print("training error: ",err / len(dataSet))

	# # test error
	# test = open('hw3test.txt', 'r')
	# testDataSet = []
	# for line in test.readlines():
	# 	line = line.strip('\n')
	# 	floats = line.split(' ')

	# 	testVec = [float(i) for i in floats[0:23]]
	# 	testDataSet.append(testVec)
	# test.close()
	# err = 0
	# index = 1
	# for item in testDataSet:
	# 	# print(index, " evaluate test error")
	# 	res = dt.predict(item)
	# 	if res != item[len(item)-1]:
	# 		err += 1
	# 	index += 1
	# print("testing error: ",err / len(dataSet))


readData()
	