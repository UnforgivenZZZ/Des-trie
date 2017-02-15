import numpy as np
import math
import sys
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
		#print(len(self.vecs[542]))
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

		return len(list(set(labs)))


class DT():

	def __init__(self,root):
		self.root = root
		self.build(self.root)

	def build(self,n):
		print(n.rule, n.bound)
		if(n.labels() == 1):
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




############################################################

def readData():
	dataSet = []
	labs = []
	f = open('hw3train.txt', 'r')
	for line in f.readlines():
		line = line.strip('\n')

		floats = line.split(' ')

		vecs = [float(i) for i in floats[0:23]]
		label = float(floats[22])

		labs.append(label)

		dataSet.append(vecs)


	f.close()
	#print(len(dataSet))

	#print(list(set(labs)))
	root = node(dataSet)
	dt = DT(root)
		


readData()
	