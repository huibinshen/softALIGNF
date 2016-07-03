import sys
import commands

import numpy as np
from numpy.linalg import inv
from scipy.io import loadmat

from sklearn.preprocessing import KernelCenterer
from sklearn import preprocessing
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from svmutil import *

classify_datasets = ['Emotions','Yeast','Enron','Fingerprint']
image_datasets= ["corel5k","espgame","iaprtc12","mirflickr"]
singlelabel_datasets = ['SPAMBASE']
bio_datasets = ['psortPos','psortNeg','plant']

datasets = singlelabel_datasets + bio_datasets + image_datasets
datasets = singlelabel_datasets

def normalize_km(K):
    D = np.diag(1/np.sqrt(np.diag(K)))
    return np.dot(np.dot(D,K),D)

def UNIMKL(n_km, n_labels):
    e = np.ones(n_km)
    w = e / np.linalg.norm(e)
    return np.tile(w[:,np.newaxis], n_labels)

def svm(train_km, test_km, train_y, test_y,tags,validate_fold):
    n_test, n_train = test_km.shape
    validate = np.array(tags == validate_fold)
    train_s = np.array(~validate)
    validate_y = train_y[validate]
    train_y_s = train_y[train_s]
    n_validate = len(validate_y)
    n_train_s = len(train_y_s)

    train_km_s = train_km[np.ix_(train_s, train_s)]
    validate_km = train_km[np.ix_(validate, train_s)]
    train_y_s = train_y[train_s]
    validate_y = train_y[validate]
    
    # convert format to use libsvm
    train_km = np.append(np.array(range(1,n_train+1)).reshape(
            n_train,1), train_km,1).tolist()
    train_km_s = np.append(np.array(range(1,n_train_s+1)).reshape(
            n_train_s,1), train_km_s,1).tolist()
    validate_km = np.append(np.array(range(1,n_validate+1)).reshape(
            n_validate,1), validate_km,1).tolist()
    test_km = np.append(np.array(range(1,n_test+1)).reshape(
            n_test,1), test_km,1).tolist()

    best_f1 = -np.inf
    best_c = 0.0001
    allcs =  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    #allcs = [1]
    for C in allcs:
        prob = svm_problem(train_y_s, train_km_s, isKernel=True)
        param = svm_parameter('-t 4 -c %f -b 0 -q' % C)
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(validate_y, validate_km,
                                            m,'-b 0 -q')
        f1 = f1_score(validate_y, p_label)
        if f1 >= best_f1:
            best_c = C
            best_f1 = f1

    C = best_c
    prob = svm_problem(train_y, train_km, isKernel=True)
    param = svm_parameter('-t 4 -c %f -b 0 -q' % C)
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(test_y, test_km, m,'-b 0 -q')
    return p_label

def svm_mkl(kernel_list, labels, mkl, n_folds, dataset, data):
    n_sample, n_labels = labels.shape
    n_km = len(kernel_list)
    tags = np.loadtxt("../data/cv/"+data+".cv")

    pred = np.zeros((n_sample, n_labels))
    # Run for each fold   
    for i in range(1,n_folds+1):
        print "Test fold %d" %i
        res_f = "../svm_result/weights/"+dataset+"_fold_%d_%s.weights" % (i,mkl)

        # divide data
        test = np.array(tags == (i+1 if i+1<6 else 1))
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

        # Normalized weights 
        normw = np.linalg.norm(wei, 2, 0)
        uni = np.ones(n_km) / np.linalg.norm(np.ones(n_km))
        for t in xrange(n_labels):
            if normw[t] == 0:  # collapsed solution
                wei[:,t] = uni
            else:
                wei[:,t] = wei[:,t] / normw[t]

        for j in range(n_labels):
            tr_y = train_y[:,j]
            te_y = test_y[:,j]
            if wei.shape[1] == 1:
                wj = wei[:,0]
            else:
                wj = wei[:,j]
            ckm = np.zeros((n_train,n_train))
            for t in range(n_km):
                ckm = ckm + wj[t]*train_km_list[t]

            # combine train and test kernel using learned weights        
            train_ckm = ckm
            test_ckm = np.zeros(test_km_list[0].shape)
            for t in range(n_km):
                test_ckm = test_ckm + wj[t]*test_km_list[t]
                
            pred_label = svm(train_ckm, test_ckm, tr_y, te_y, tags[train], i)
            pred[test, j] = pred_label
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
                km_list.append(normalize_km(np.loadtxt(km_f)))
            pred_f = "../svm_result/pred/%s_cvpred_%s.txt" % (data, mkl)
            pred = svm_mkl(km_list, y, mkl, 5, data,data)
            np.savetxt(pred_f, pred, fmt="%d")

        elif data == 'plant' or data == 'psortPos' or data == 'psortNeg':
            y = loadmat(km_dir+"label_%s.mat" % data)['y'].ravel()
            km_list = []
            fs = commands.getoutput('ls %skern\;substr*.mat' % km_dir).split("\n")
            for f in fs:
                km = loadmat(f)
                km_list.append(km['K'])
            fs = commands.getoutput('ls %skern\;phylpro*.mat' % km_dir).split("\n")
            for f in fs:
                km = loadmat(f)
                km_list.append(km['K'])
            fs = commands.getoutput('ls %skm_evalue*.mat' % km_dir).split("\n")
            for f in fs:
                km = loadmat(f)
                km_list.append(km['K'])

            n_samples = y.shape[0]
            n_km = len(km_list)
            y_pred = np.zeros(n_samples)
            n_labels = 1
            tags = np.loadtxt("../data/cv/"+data+".cv")
            for fold in range(1,6):
                test_ind = np.where(tags == fold)[0]
                train_ind = np.where(tags != fold)[0]
                train_km_list = []
                test_km_list = []
                train_y = y[train_ind]
                test_y = y[test_ind]
                n_train = len(train_ind)
                n_test = len(test_ind)

                w_f = "../svm_result/weights/"+data+"_fold_%d_%s.weights" % (fold,mkl)

                if mkl == 'UNIMKL':
                    w = UNIMKL(n_km, n_labels).ravel()
                else:
                    w = np.loadtxt(w_f, ndmin=2).ravel()

                normw = np.linalg.norm(w, 2, 0)
                uni = np.ones(n_km) / np.linalg.norm(np.ones(n_km))
                if normw == 0:
                    w = uni
                else:
                    w = w / normw

                for km in km_list:
                    kc = KernelCenterer()
                    train_km = km[np.ix_(train_ind, train_ind)]
                    test_km = km[np.ix_(test_ind, train_ind)]
                    # center train and test kernels                      
                    kc.fit(train_km)
                    train_km_c = kc.transform(train_km)
                    test_km_c = kc.transform(test_km)
                    train_km_list.append(train_km_c)
                    test_km_list.append(test_km_c)

                train_ckm = np.zeros((n_train,n_train))
                for t in range(n_km):
                    train_ckm = train_ckm + w[t]*train_km_list[t]
                test_ckm = np.zeros(test_km_list[0].shape)
                for t in range(n_km):
                    test_ckm = test_ckm + w[t]*test_km_list[t]
                
                C_range = [0.01,0.1,1,10,100]
                param_grid = dict(C=C_range)
                cv = StratifiedShuffleSplit(train_y,n_iter=5,test_size=0.2,random_state=42)
                grid = GridSearchCV(SVC(kernel='precomputed'), param_grid=param_grid, cv=cv)
                grid.fit(train_ckm, train_y)
                bestC = grid.best_params_['C']
                svm = SVC(kernel='precomputed', C=bestC)
                svm.fit(train_ckm, train_y)
                y_pred[test_ind] = svm.predict(test_ckm)

            pred_f = "../svm_result/pred/%s_cvpred_%s.txt" % (data, mkl)
            np.savetxt(pred_f, y_pred, fmt="%d")

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
                linear_km_list.append(normalize_km(km))
            pred_f = "../svm_result/pred/%s_cvpred_%s.txt" % (data, mkl)
            pred = svm_mkl(linear_km_list, y, mkl, 5, data,data)
            np.savetxt(pred_f, pred, fmt="%d")

        elif data == 'SPAMBASE':
            y = np.loadtxt(km_dir+"y.txt",ndmin=2)
            rbf_km_list = []
            gammas = [2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3]
            X = np.loadtxt(km_dir+"x.txt")
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)
            X = preprocessing.normalize(X)
            for gamma in gammas:
                km = rbf_kernel(X, gamma=gamma)
                rbf_km_list.append(km)
            pred_f = "../svm_result/pred/%s_cvpred_%s.txt" % (data, mkl)
            pred = svm_mkl(rbf_km_list, y, mkl, 5, data,data)
            np.savetxt(pred_f, pred, fmt="%d")

        else:
            rbf_km_list = []
            gammas = [2**-13,2**-11,2**-9,2**-7,2**-5,2**-3,2**-1,2**1,2**3]
            X = np.loadtxt(km_dir+"x.txt")
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)
            X = preprocessing.normalize(X)
            y = np.loadtxt(km_dir+"y.txt")
            p = np.sum(y==1,0)/float(y.shape[0])        
            y = y[:,p>t]
            for gamma in gammas:
                km = rbf_kernel(X, gamma=gamma)
                # normalize input kernel !!!!!!!!
                rbf_km_list.append(km)
            pred_f = "../svm_result/pred/%s_cvpred_%s.txt" % (data, mkl)
            pred = svm_mkl(rbf_km_list, y, mkl, 5, data,data)
            np.savetxt(pred_f, pred, fmt="%d")

#cls('UNIMKL')
#cls('ALIGNF2SOFT')
cls('ALIGNF2')
#cls('TSMKL')

