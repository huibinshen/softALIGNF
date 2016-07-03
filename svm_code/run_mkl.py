import commands
import numpy as np

from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from alignf import ALIGNF
from alignfSLACK import ALIGNFSLACK
from ovkr import OVKR_train, OVKR_test
from tsmkl import TSMKL

classify_datasets = ['Emotions','Yeast','Enron','Fingerprint']
image_datasets= ["corel5k","espgame","iaprtc12","mirflickr"]
singlelabel_datasets = ['SPAMBASE']
bio_datasets = ['psortPos','psortNeg','plant']

datasets = singlelabel_datasets + bio_datasets + image_datasets
datasets = singlelabel_datasets

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

def UNIMKL(km_list, Y):
    n_km = len(km_list)
    n_samples, n_labels = Y.shape
    e = np.ones(n_km)
    w = e / np.linalg.norm(e)
    return np.tile(w[:,np.newaxis], n_labels)

def alignf2(kernel_list, Y, data):
    """ALIGNF independently for each label"""
    n_samples, n_labels = Y.shape
    n_kernels = len(kernel_list)
    res = np.zeros((n_kernels, n_labels))
    for t in range(n_labels):
        y = Y[:,t].reshape(n_samples,1)
        if data in bio_datasets:
            ky = np.zeros((n_samples, n_samples))
            for i in xrange(n_samples):
                for j in xrange(i, n_samples):
                    if y[i] == y[j]:
                        ky[i,j] = 1
                    else:
                        ky[i,j] = -1
                ky[j,i] = ky[i,j]
            ky = normalize_km(ky)
        else:
            ky = normalize_km(np.dot(y, y.T))
        w = ALIGNF(kernel_list, ky)
        res[:,t] = w
    return res

def svm(train_km, test_km, train_y, test_y):
    # train on train_km and train_y, predict on test_km
    # return prediction
    para_grid ={'C':[1e-2, 1e-1, 1, 10, 100]}
    svc = SVC(kernel='precomputed')
    clf = grid_search.GridSearchCV(svc, para_grid)
    clf.fit(train_km, train_y)
    pred = clf.predict(test_km)
    # for multi-class task, use acc other than F1
    #return f1_score(test_y, pred)
    return accuracy_score(test_y, pred)

def ALIGNFSOFT(kernel_list, ky, y, test_fold, tags):
    # Find best upper bound in CV and train on whole data
    # Reutrn the weights 
    y = y.ravel()
    n_km = len(kernel_list)

    tag = np.array(tags)
    tag = tag[tag!=test_fold]
    remain_fold = np.unique(tag).tolist()
    all_best_c = []
    for validate_fold in remain_fold:
        train = tag != validate_fold
        validate = tag == validate_fold
        # train on train fold ,validate on validate_fold.
        # Do not use test fold. test fold used in outter cv
        ky_train = ky[np.ix_(train, train)]
        y_train = y[train]
        y_validate = y[validate]
        train_km_list = []
        validate_km_list = []
        n_train = len(y_train)
        n_validate = len(y_validate)

        for km in kernel_list:
            kc = KernelCenterer()
            train_km = km[np.ix_(train, train)]
            validate_km = km[np.ix_(validate, train)]
            # center train and validate kernels                      
            train_km_c = kc.fit_transform(train_km)
            train_km_list.append(train_km_c)
            validate_km_c = kc.transform(validate_km)
            validate_km_list.append(validate_km_c)

        # if the label is too biased, SVM CV will fail, just return ALIGNF solution
        if np.sum(y_train==1) > n_train-3 or np.sum(y_train==-1) > n_train-3:
            return 1e8, ALIGNFSLACK(train_km_list, ky_train, 1e8) 

        Cs = np.exp2(np.array(range(-9,7))).tolist() + [1e8]
        W = np.zeros((n_km, len(Cs)))
        for i in xrange(len(Cs)):
            W[:,i] = ALIGNFSLACK(train_km_list, ky_train, Cs[i])

        W = W / np.linalg.norm(W, 2, 0)
        f1 = np.zeros(len(Cs))
        for i in xrange(len(Cs)):
            train_ckm = np.zeros((n_train,n_train))
            validate_ckm = np.zeros((n_validate,n_train))
            w = W[:,i]
            for j in xrange(n_km):
                train_ckm += w[j]*train_km_list[j]
                validate_ckm += w[j]*validate_km_list[j]
            f1[i] = svm(train_ckm, validate_ckm, y_train, y_validate)
        # return the first maximum
        maxind = np.argmax(f1)
        bestC = Cs[maxind]
        all_best_c.append(bestC)
        print f1
        print "..Best C is", bestC

    bestC = np.mean(all_best_c)
    print "..Take the average best upper bound", bestC
    # use the best upper bound to solve ALIGNFSOFT
    return bestC, ALIGNFSLACK(kernel_list, ky, bestC)    

def ALIGNF2SOFT(kernel_list, Y, test_fold, tags, data):
    """ALIGNFSOFT independently, find best C within innner cv, then train on whole"""
    n_samples, n_labels = Y.shape
    n_kernels = len(kernel_list)
    res = np.zeros((n_kernels, n_labels))
    bestC = np.zeros(n_labels)
    for t in xrange(n_labels):
        print ".. Label",t
        y = Y[:,t].reshape(n_samples,1)
        if data == 'plant' or data == 'psortPos' or data == 'psortNeg':
            ky = np.zeros((n_samples, n_samples))
            for i in xrange(n_samples):
                for j in xrange(i, n_samples):
                    if y[i] == y[j]:
                        ky[i,j] = 1
                    else:
                        ky[i,j] = -1
                ky[j,i] = ky[i,j]
            ky = normalize_km(ky)
        else:
            ky = normalize_km(np.dot(y, y.T))        
        bestC[t], res[:,t] = ALIGNFSOFT(kernel_list, ky, y, test_fold, tags)
    return bestC, res

def cv_mkl(kernel_list, labels, mkl, n_folds, dataset, data):

    n_sample, n_labels = labels.shape
    n_km = len(kernel_list)
    tags = np.loadtxt("../data/cv/"+data+".cv")

    for i in range(1,n_folds+1):
        print "Test fold %d" %i
        res_f = "../svm_result/weights/"+dataset+"_fold_%d_%s.weights" % (i,mkl)
        para_f = "../svm_result/upperbound/"+dataset+"_fold_%d_%s.ubound" % (i,mkl)
        test = np.array(tags == i)
        train = np.array(~test)
        train_y = labels[train,:]
        test_y = labels[test,:]
        n_train = len(train_y)
        n_test = len(test_y)
        train_km_list = []

        # all train kernels are nomalized and centered
        for km in kernel_list:
            kc = KernelCenterer()
            train_km = km[np.ix_(train, train)]
            # center train and test kernels                      
            kc.fit(train_km)
            train_km_c = kc.transform(train_km)
            train_km_list.append(train_km_c)

        if mkl == 'UNIMKL':
            res = UNIMKL(train_km_list, train_y)
            np.savetxt(res_f, res)            
        if mkl == 'ALIGNF2':
            res = alignf2(train_km_list, train_y, data)
            np.savetxt(res_f, res)
        if mkl.find('ALIGNF2SOFT') != -1:
            bestC, res = ALIGNF2SOFT(train_km_list, train_y, i, tags, data)
            np.savetxt(res_f, res)
            np.savetxt(para_f, bestC)
        if mkl == "TSMKL":
            W = np.zeros((n_km, n_labels))
            for j in xrange(n_labels):
                print "..label",j
                W[:,j] = TSMKL(train_km_list, train_y[:,j])
            res_f = "../svm_result/weights/"+dataset+"_fold_%d_%s.weights" % (i,mkl)
            np.savetxt(res_f, W)
            
def cls(mkl):
    for data in datasets:
        print "####################"
        print '# ',data
        print "####################" 
        # consider labels with more than 2% positive examples
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
            cv_mkl(km_list, y, mkl, 5, data, data)

        elif data == 'plant' or data == 'psortPos' or data == 'psortNeg':
            y = loadmat(km_dir+"label_%s.mat" % data)['y']
            km_list = []
            fs = commands.getoutput('ls %skern\;substr*.mat' % datadir).split("\n")
            for f in fs:
                km = loadmat(f)
                km_list.append(km['K'])
            fs = commands.getoutput('ls %skern\;phylpro*.mat' % datadir).split("\n")
            for f in fs:
                km = loadmat(f)
                km_list.append(km['K'])
            fs = commands.getoutput('ls %skm_evalue*.mat' % datadir).split("\n")
            for f in fs:
                km = loadmat(f)
                km_list.append(km['K'])
            cv_mkl(km_list, y, mkl, 5, data, data)

        elif data in image_datasets:
            y = np.loadtxt(km_dir+"y.txt",ndmin=2)
            p = np.sum(y==1,0)/float(y.shape[0])        
            y = y[:,p>t]
            linear_km_list = []
            for i in range(1,16):
                name = '/kernel_linear_%d.txt' % i
                km_f = km_dir+name
                km = np.loadtxt(km_f)
                linear_km_list.append(normalize_km(km))
            cv_mkl(linear_km_list, y, mkl, 5, data, data)

        elif data == 'SPAMBASE':
            y = np.loadtxt(km_dir+"y.txt",ndmin=2)
            rbf_km_list = []
            gammas = [2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3]
            X = np.loadtxt(km_dir+"/x.txt")
            scaler = preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)
            X = preprocessing.normalize(X)
            for gamma in gammas:
                km = rbf_kernel(X, gamma=gamma)
                rbf_km_list.append(km)
            cv_mkl(rbf_km_list, y, mkl, 5, data,data)
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
                rbf_km_list.append(km)
            cv_mkl(rbf_km_list, y, mkl, 5, data,data)

#cls("UNIMKL")
#cls('ALIGNF2SOFT')
cls('ALIGNF2')
#cls('TSMKL')
