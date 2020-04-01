import numpy as np


from opfython.models.unsupervised import UnsupervisedOPF
from opfython.models.supervised import SupervisedOPF
from O2PF_oversampling import O2PF

import sys
import logging
logging.disable(sys.maxsize)



#configurations
datasets = ['indian_liver','secom','seismic_bumps', 'spam','vertebral_column','wilt']

files = ['major_negative','major_neutral','negative','negatives_major_zero']
folds = np.arange(1,21)

percents = [1.0]
k_max = [5,10,20,30,40,50]
o2pf = O2PF('ResultsGeneral')

#Running
for fl in files:
    for dsds in range(len(datasets)):
        ds = datasets[dsds]

        for ff in range(len(folds)): 
            f = folds[ff]
            o2pf.run(ds, f, percents,k_max,2,fl)
