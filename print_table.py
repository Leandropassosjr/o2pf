import numpy as np
from scipy.stats import wilcoxon
import os

class PT(object):

	def __init__(self, datasets,algorithms,percentOPF,percentSMOTE,percentLabel,resultsFolder):
		self.datasets=datasets
		self.algorithms=algorithms 
		self.percentOPF =percentOPF
		self.percentSMOTE=percentSMOTE
		self.percentLabel=percentLabel
		self.resultsFolder=resultsFolder

	def __loadOPF(self, ds,p, alg):
		X = []

		for i in np.arange(1,21):
		    val = np.loadtxt('{}/{}/{}/{}/{}/results.txt'.format(self.resultsFolder,self.algorithms[alg],self.datasets[ds],str(i),self.percentOPF[p]),delimiter=',')
		    
		    X.append(val)
		
		X = np.asarray(np.vstack(X))
		return X

	def __loadDown(self, ds, alg):
		X = []

		for i in np.arange(1,21):
		    val = np.loadtxt('{}/{}/{}/{}/results.txt'.format(self.resultsFolder,self.algorithms[alg],self.datasets[ds],str(i)),delimiter=',')
		    
		    X.append(val)
		
		X = np.asarray(np.vstack(X))
		return X

	def __loadFelix(self, ds, alg):
		X = np.zeros((20,4))
		val = np.loadtxt('{}/{}/{}/result'.format(self.resultsFolder,self.algorithms[alg],self.datasets[ds]),delimiter=',')
		

		X[:,0] = val[5,:]
		X[:,1] = val[0,:]
		X[:,2] = val[2,:]
		X[:,3] = val[3,:]

		return X

	def __loadSMOTE(self,ds,p, alg):
		X = []
		for i in np.arange(1,21):
		    path = '{}/{}/{}/{}/{}/results_test.txt'.format(self.resultsFolder,self.algorithms[alg],self.datasets[ds],str(i),self.percentOPF[p]) 
		    if not os.path.exists(path):
		        path = '{}/{}/{}/{}/{}/results_test.txt'.format(self.resultsFolder,self.algorithms[alg],self.datasets[ds],str(i),self.percentSMOTE[p]) 
		    
		    val = np.loadtxt(path,delimiter=' ',skiprows=1)
		    X.append(val)
		
		X = np.asarray(np.vstack(X))
		return X

	def __loadDownBaseline(self,ds, alg):
		X = []
		for i in np.arange(1,21):
		    path = '{}/{}/{}/{}/results_best.txt'.format(self.resultsFolder,self.algorithms[alg],self.datasets[ds],str(i)) 

		    val = np.loadtxt(path,delimiter=' ',skiprows=1)
		    X.append(val)
		
		X = np.asarray(np.vstack(X))
		return X


	def __loadOriginal(self,ds, alg):
		X = []
		for i in np.arange(1,21):
		    if self.algorithms[alg] =='ORIGINAL':
		        path = '{}/{}/{}/{}/results.tex'.format(self.resultsFolder,self.algorithms[alg],self.datasets[ds],str(i)) 
		    else:
		        path = '{}/{}/{}/{}/results.txt'.format(self.resultsFolder,self.algorithms[alg],self.datasets[ds],str(i)) 
		    val = np.loadtxt(path,delimiter=',')
		    X.append(val)
		
		X = np.asarray(np.vstack(X))
		return X

	def calcularValores(self, ds):
		# # algorithms, # statistics
		mat = np.zeros((len(self.algorithms),len(self.percentLabel),8))

		for alg in range(len(self.algorithms)):
			for p in range(len(self.percentLabel)):        
				if self.algorithms[alg] in ['OPF','OVER_DOWN_major_negative','OVER_DOWN_major_neutral','OVER_DOWN_negative','OVER_DOWN_negatives_major_zero']:            
					X = self.__loadOPF(ds,p,alg)
				elif self.algorithms[alg] =='ORIGINAL':
					X = self.__loadOriginal(ds,alg)
				elif self.algorithms[alg] in ['mean_interp','mean_mahalanobis','mean_geo','mean_geo_mb','dist','dist_interp', 'geometric_geo_mb','proto','proto_interp','weight','weight_interp']:
					X = self.__loadFelix(ds,alg)
				elif self.algorithms[alg] in ['down_major_negative','down_major_neutral','down_negative','down_negatives_major_zero','down_balance']:
					X = self.__loadDown(ds,alg)
				elif self.algorithms[alg] in ['CNN','NearMiss-1','NearMiss-2','NearMiss-3']:
					X = self.__loadDownBaseline(ds,alg)
				else: 
					X = self.__loadSMOTE(ds,p,alg)               
				mat[alg,p,0] = np.average(X[:,0])#best k
				mat[alg,p,1] = np.std(X[:,0])#std Best k
				mat[alg,p,2] = np.average(X[:,1])#accuracy
				mat[alg,p,3] = np.std(X[:,1])#std accuracy
				mat[alg,p,4] = np.average(X[:,2])#recall
				mat[alg,p,5] = np.std(X[:,2])#std recall
				mat[alg,p,6] = np.average(X[:,3])#f1
				mat[alg,p,7] = np.std(X[:,3])#std f1

		return mat

	def calcularWilcoxon(self, ds,p, mat, index_metric=3):
		#index_metric: 0 = k/k_max, 1 = Accuracy, 2 = Recall, 3 = F1
		# # algorithms, # metaheuristic tech
		wil = np.zeros(len(self.algorithms))


		alg_best= mat.argmax()

		wil[alg_best] = 1
		if self.algorithms[alg_best] in ['OPF','OVER_DOWN_major_negative','OVER_DOWN_major_neutral','OVER_DOWN_negative','OVER_DOWN_negatives_major_zero']:
			better =self.__loadOPF(ds,p, alg_best)
		elif self.algorithms[alg_best] =='ORIGINAL':
			better = self.__loadOriginal(ds,alg_best)

		elif self.algorithms[alg_best] in ['mean_interp','mean_mahalanobis','mean_geo','mean_geo_mb','dist','dist_interp', 'geometric_geo_mb','proto','proto_interp','weight','weight_interp']:
			better = self.__loadFelix(ds,alg_best)
		elif self.algorithms[alg_best] in ['down_major_negative','down_major_neutral','down_negative','down_negatives_major_zero','down_balance']:
			better = self.__loadDown(ds,alg_best)
		elif self.algorithms[alg_best] in ['CNN','NearMiss-1','NearMiss-2','NearMiss-3']:
			better = self.__loadDownBaseline(ds,alg_best)
		else:
			better = self.__loadSMOTE(ds,p, alg_best)    
		better = better[:,2]
		for alg in range(len(self.algorithms)):
			if alg_best !=alg:
				if self.algorithms[alg] in ['OPF','OVER_DOWN_major_negative','OVER_DOWN_major_neutral','OVER_DOWN_negative','OVER_DOWN_negatives_major_zero']:
					x =self.__loadOPF(ds,p, alg)
				elif self.algorithms[alg] =='ORIGINAL':
					x = self.__loadOriginal(ds,alg)
				elif self.algorithms[alg] in ['mean_interp','mean_mahalanobis','mean_geo','mean_geo_mb','dist','dist_interp', 'geometric_geo_mb','proto','proto_interp','weight','weight_interp']:
					x = self.__loadFelix(ds,alg)
				elif self.algorithms[alg] in ['down_major_negative','down_major_neutral','down_negative','down_negatives_major_zero','down_balance']:
					x = self.__loadDown(ds,alg)
				elif self.algorithms[alg] in ['CNN','NearMiss-1','NearMiss-2','NearMiss-3']:
					x = self.__loadDownBaseline(ds,alg)
				else:
					x = self.__loadSMOTE(ds,p, alg)    
		        
				statistic, pvalue = wilcoxon(better,x[:,index_metric])
				if pvalue>=0.05:
					wil[alg] = 1

		return wil
