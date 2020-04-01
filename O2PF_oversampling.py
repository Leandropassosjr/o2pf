import numpy as np
from opfython.models.unsupervised import UnsupervisedOPF
from opfython.models.supervised import SupervisedOPF
from sklearn.metrics import accuracy_score, recall_score, f1_score
import time
import os

class O2PF(object):

	def __init__(self, path_output):
		self.opfSup = SupervisedOPF(distance='log_squared_euclidean', pre_computed_distance=None)
		self.path_output=path_output

	def __generateSamples(self, n_samples, mean, cov):
		epsilon = 0.0001
		d = len(mean)
		K = cov + epsilon*np.identity(d)
		
		z = np.random.multivariate_normal(mean=mean.reshape(d,), cov=K, size=n_samples)

		return z


	def __getMeanCov(self, kmax, data, labels):   
		classes = np.unique(labels)

		max_class = 0
		max_class_idx = -1
		dt = []
		lbl = []

		for c in range(len(classes)):
		    idx = labels == classes[c]
		    dt.append(data[idx].copy())
		    lbl.append(labels[idx].copy())
		    if max_class<len(dt[c]):
		        max_class_idx = c
		        max_class = len(dt[c]) 
		
		out = {}

		for c in range(len(classes)):
		    if c == max_class_idx:
		        out[c] = None
		        continue

		    train_data = dt[c]
		    train_label = lbl[c]
		    
		    opf = UnsupervisedOPF(min_k=1, max_k=kmax, distance='log_squared_euclidean', pre_computed_distance=None)
		    
		    opf.fit(train_data,train_label)
		    preds=[]
		    for i in range((opf.subgraph.n_nodes)):
		        preds.append(opf.subgraph.nodes[i].cluster_label)
		    #print(preds)

		    c_map = {}
		    preds = np.asarray(preds)

		    for cl in range(opf.subgraph.n_clusters):
		        idx_c = preds == cl
		        cluster = train_data[idx_c]
		        mean = cluster.mean(0)
		        cov = np.cov(cluster, rowvar=False)

		        if cov.size!=1:
		            c_map[cl] = (mean, cov, len(cluster))
		        else:
		            c_map[cl] = None
		    out[c] = (c_map, opf.subgraph.n_clusters,len(train_label) )
		return out, len(classes), classes, max_class

	def __O2PF(self,k,f, p,data, labels,trainSet):
		start_time = time.time()
		    
		values, n_classes, classes, max_class = self.__getMeanCov(k,data,labels)


		finalSamples = []  

		for i in range(n_classes):
		    if values[i] is None:
		        continue
		    values_ = values[i][0]
		    n_clusters = values[i][1]
		    classe = classes[i]
		    newSamples = []


		    #number of samples of class classe
		    samplesPerClass = values[i][2]  
		    #total number of samples to be generated to class classe
		    NumberOfSamples = max_class-samplesPerClass

		    #number of samples that will be actually generated to class classe
		    NumberOfSamplesToGenerate = int(NumberOfSamples*p)


		    for j in range(n_clusters):               
		        if values_[j] is None:
		            continue
		        mean = values_[j][0]
		        cov = values_[j][1]
		        #number of samples from cluster j
		        n_samples = values_[j][2]  

		        #number of samples generated based on cluster j
		        samplesToGenerate = int((NumberOfSamplesToGenerate* n_samples)/samplesPerClass)                

		        newSamples.append(self.__generateSamples(samplesToGenerate, mean,cov))
		    
		    if len(newSamples[0])==0:
		        continue
		    
		    newSamples = np.asarray(np.vstack(newSamples))
		    newSamples = np.insert(newSamples,len(newSamples[0]),classe , axis=1)            

		    finalSamples.append(newSamples)
		    
		finalSamples.append(trainSet)
		finalSamples = np.asarray(np.vstack(finalSamples))


		    
		end_time = time.time()
		return finalSamples[:,:-1], finalSamples[:,-1].astype(np.int)  , end_time-start_time

	def __classify(self, x_train,y_train, x_valid, y_valid, minority_class):
		# Training the OPF                
		self.opfSup.fit(x_train, y_train)

		# Prediction of the validation samples
		y_pred = self.opfSup.predict(x_valid)
		y_pred = np.array(y_pred)

		# Validation measures for this k nearest neighbors
		accuracy = accuracy_score(y_valid, y_pred)
		recall = recall_score(y_valid, y_pred, pos_label=minority_class) # assuming that 2 is the minority class
		f1 = f1_score(y_valid, y_pred, pos_label=minority_class)

		return accuracy, recall, f1, y_pred

	def run(self, ds, f, percents,k_max, minority_class,variant=''):
		'''
			variant: algorithm variant
			ds: dataset
			f: fold
		'''

		if variant=='':
			train = np.loadtxt('data/{}/{}/train.txt'.format(ds,f),delimiter=',', dtype=np.float32)
		else:
			train = np.loadtxt('data/{}/{}/train_{}.txt'.format(ds,f,variant),delimiter=',', dtype=np.float32)
		valid = np.loadtxt('data/{}/{}/valid.txt'.format(ds,f),delimiter=',', dtype=np.float32)
		test = np.loadtxt('data/{}/{}/test.txt'.format(ds,f),delimiter=',', dtype=np.float32)
		data = train[:,:-1]
		labels = train[:,-1].astype(np.int)  
		fmt = '%.5f,'*(len(train[0])-1)+'%.d'

		x_valid = valid[:,:-1]
		y_valid = valid[:,-1].astype(np.int)  

		x_test = test[:,:-1]
		y_test = test[:,-1].astype(np.int)  

		for pp in range(len(percents)): 
			p = percents[pp]

			if variant=='':
				path = '{}/OPF/{}/{}/{}'.format(self.path_output,ds,f,p)
			else:
				path = '{}/OVER_DOWN_{}/{}/{}/{}'.format(self.path_output,variant,ds,f,p)
			if not os.path.exists(path):
				os.makedirs(path)

			validation_print=[]
			results_print=[]

			k_best = 0
			best_recall = 0
			for kk in range(len(k_max)): 
				k = k_max[kk]
				x_train, y_train, elapsed_time = self.__O2PF(k,f, p,data,labels, train)                 
				accuracy, recall, f1, y_pred= self.__classify(x_train,y_train, x_valid, y_valid,minority_class)
				np.savetxt('{}/pred_validation_{}.txt'.format(path,k), y_pred, fmt='%d')

				validation_print.append([k, accuracy, recall, f1,elapsed_time])

				if recall>best_recall:
					best_recall = recall
					k_best = k

			x_train, y_train, elapsed_time = self.__O2PF(k_best,f, p,data,labels, train)  
			accuracy, recall, f1, y_pred = self.__classify(x_train,y_train, x_test, y_test,minority_class)
			np.savetxt('{}/pred_test.txt'.format(path), y_pred, fmt='%d')


			results_print.append([k_best,accuracy, recall, f1,elapsed_time])


			np.savetxt('{}/{}.txt'.format(path,'validation'), validation_print, fmt='%d,%.5f,%.5f,%.5f,%.5f')
			np.savetxt('{}/{}.txt'.format(path,'results'), results_print, fmt='%d,%.5f,%.5f,%.5f,%.5f')

