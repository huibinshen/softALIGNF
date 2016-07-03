import sys
import commands

import numpy as np
from scipy.io import loadmat

from sklearn.preprocessing import KernelCenterer
from sklearn import preprocessing
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from ovkr import OVKR_train, OVKR_test, OVKR_train_CV

# OVKR is multi-label methods, do not run on bio_datasets and singlelabel_datasets
classify_datasets = ['Emotions','Yeast', 'Enron', 'Fingerprint']
image_datasets= ["corel5k","espgame","iaprtc12","mirflickr"]
bio_datasets = ['psortPos','psortNeg','plant']

datasets = classify_datasets + image_datasets
datasets = ["Emotions"]


def normalize_km(K):
    D = np.diag(1/np.sqrt(np.diag(K)))
    return np.dot(np.dot(D,K),D)

def center(km):
    """ centering km """
    m = len(km)
    I = np.eye(m)
    one = np.ones((m,1))
    t = I - np.dot(one,one.T)/m
    return np.dot(np.dot(t,km),t)

def UNIMKL(n_km, n_labels):
    e = np.ones(n_km)
    w = e / np.linalg.norm(e)
    return np.tile(w[:,np.newaxis], n_labels)

def ovkr_mkl(kernel_list, labels, mkl, n_folds, dataset, data):
    n_sample, n_labels = labels.shape
    n_km = len(kernel_list)
    tags = np.loadtxt("../data/cv/"+data+".cv")
    #tags = np.array(range(n_sample)) % n_folds + 1
    #np.random.seed(1234)
    #np.random.shuffle(tags)

    pred = np.zeros((n_sample, n_labels))

    # Run for each fold   
    for i in range(1,n_folds+1):
        print "Test fold %d" %i
        res_f = "../ovkr_result/weights/"+dataset+"_fold_%d_%s.weights" % (i,mkl)

        # divide data
        test = np.array(tags == i)
        train = np.array(~test)
        train_y = labels[train,:]
        test_y = labels[test,:]
        n_train = len(train_y)
        n_test = len(test_y)

        train_km_list = []
        test_km_list = []
        for km in kernel_list:
            kc = KernelCenterer()
            train_km = km[np.ix_(train, train)]
            test_km = km[np.ix_(test, train)]
            # center train and test kernels                      
            kc.fit(train_km)
            train_km_c = kc.transform(train_km)
            test_km_c = kc.transform(test_km)
            train_km_list.append(train_km_c)
            test_km_list.append(test_km_c)

        if mkl == 'UNIMKL':
            wei = UNIMKL(n_km, n_labels)
        else:
            wei = np.loadtxt(res_f, ndmin=2)        

        normw = np.linalg.norm(wei)
        uni = np.ones(n_km) / np.linalg.norm(np.ones(n_km))
        if normw == 0:
            wei[:,0] = uni
        else:
            wei[:,0] = wei[:,0] / normw

        train_ckm = np.zeros((n_train,n_train))
        for t in range(n_km):
            train_ckm += wei[t,0]*train_km_list[t]

        # combine train and test kernel using learned weights        
        test_ckm = np.zeros(test_km_list[0].shape)
        for t in range(n_km):
            test_ckm = test_ckm + wei[t,0]*test_km_list[t]
                
        AP = OVKR_train_CV(train_ckm, train_y, tags[train])
        pred_label = OVKR_test(test_ckm, AP)
        pred[test, :] = pred_label
    return pred

def cls(mkl):

    for data in datasets:
        print "####################"
        print '# ',data
        print "####################" 
        # consider labels with more than 2%
        t = 0.02
        datadir = '../data/'
        km_dir = datadir + data + "/"
        if data == 'Fingerprint':
            kernels = ['PPKr', 'NB','CP2','NI','LB','CPC','RLB','LC','LI','CPK','RLI','CSC']
            km_list = []
            y = np.loadtxt(km_dir+"y.txt",ndmin=2)
            p = np.sum(y==1,0)/float(y.shape[0])        
            y = y[:,p>t]

            for k in kernels:
                km_f = datadir + data + ("/%s.txt" % k)
                km_list.append(center(normalize_km(np.loadtxt(km_f))))

            pred_f = "../ovkr_result/pred/%s_cvpred_%s.npy" % (data, mkl)
            pred = ovkr_mkl(km_list, y, mkl, 5, data,data)
            np.save(pred_f, pred)

        elif data in image_datasets:
            y = np.loadtxt(km_dir+"y.txt",ndmin=2)
            p = np.sum(y==1,0)/float(y.shape[0])        
            y = y[:,p>t]
            linear_km_list = []
            for i in range(1,16):
                name = 'kernel_linear_%d.txt' % i
                km_f = km_dir+name
                km = np.loadtxt(km_f)
                # normalize input kernel !!!!!!!!
                linear_km_list.append(center(normalize_km(km)))
            pred_f = "../ovkr_result/pred/%s_cvpred_%s.npy" % (data, mkl)
            pred = ovkr_mkl(linear_km_list, y, mkl, 5, data,data)
            np.save(pred_f, pred)

        else:
            rbf_km_list = []
            gammas = [2**-13,2**-11,2**-9,2**-7,2**-5,2**-3,2**-1,2**1,2**3]
            X = np.loadtxt(km_dir+"/x.txt")
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)
            X = preprocessing.normalize(X)
            y = np.loadtxt(km_dir+"y.txt")
            p = np.sum(y==1,0)/float(y.shape[0])        
            y = y[:,p>t]
            for gamma in gammas:
                km = rbf_kernel(X, gamma=gamma)
                # normalize input kernel !!!!!!!!
                rbf_km_list.append(center(km))
            pred_f = "../ovkr_result/pred/%s_cvpred_%s.npy" % (data, mkl)
            pred = ovkr_mkl(rbf_km_list, y, mkl, 5, data,data)
            np.save(pred_f, pred)


#cls('ALIGNFSOFT')
cls('ALIGNF')
#cls('UNIMKL')
