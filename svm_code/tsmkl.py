import sys
import numpy as np
sys.path.append('pegasos')

from sklearn.cross_validation import train_test_split

import pegasos

def TSMKL(km_list, y):
    n_km = len(km_list)
    n_samples = km_list[0].shape[0]
    y = y.reshape(n_samples, 1)

    if len(np.unique(y)) > 2:
        ky = np.zeros((n_samples, n_samples))
        for i in xrange(n_samples):
            for j in xrange(i, n_samples):
                if y[i] == y[j]:
                    ky[i,j] = 1
                else:
                    ky[i,j] = -1
                ky[j,i] = ky[i,j]
    else:
        ky = np.dot(y,y.T)
        
    # get the upper diagnoal indices
    iu = np.triu_indices(n_samples)
    
    XX = []
    yy = ky[iu].ravel()
    for km in km_list:
        XX.append(km[iu].ravel())
    XX = np.array(XX).T

    if np.sum(yy==1) == yy.shape[0] or np.sum(yy==-1) == yy.shape[0]:
        print "extemely biased label, setting to uni weights"
        uni = np.ones(n_km) / np.linalg.norm(np.ones(n_km))
        return uni

    datasets = train_test_split(XX, yy, random_state=12345)
    train_X, test_X, train_y, test_y = datasets    
    if np.sum(train_y==1) == train_y.shape[0] or np.sum(train_y==-1) == train_y.shape[0]:
        uni = np.ones(n_km) / np.linalg.norm(np.ones(n_km))
        print "extemely biased label, setting to uni weights"
        return uni

    lambds = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2]
    scores = []
    for lambd in lambds:
        model = pegasos.PegasosSVMClassifier(lambda_reg=lambd)
        model.fit(train_X, train_y)
        score = model.score(test_X, test_y)
        scores.append(score)

    bestlambda = lambds[np.argmax(scores)]
    model = pegasos.PegasosSVMClassifier(lambda_reg=bestlambda)
    model.fit(train_X, train_y)
    #print scores
    print ".. best lambda", bestlambda
    print "..",model.weight_vector.weights
    return model.weight_vector.weights


def TSMKLKY(km_list, ky):
    n_km = len(km_list)
    n_samples = km_list[0].shape[0]

    # get the upper diagnoal indices
    iu = np.triu_indices(n_samples)
    
    XX = []
    yy = ky[iu].ravel()
    for km in km_list:
        XX.append(km[iu].ravel())
    XX = np.array(XX).T

    if np.sum(yy==1) == yy.shape[0] or np.sum(yy==-1) == yy.shape[0]:
        print "extemely biased label, setting to uni weights"
        uni = np.ones(n_km) / np.linalg.norm(np.ones(n_km))
        return uni

    datasets = train_test_split(XX, yy, random_state=12345)
    train_X, test_X, train_y, test_y = datasets    
    if np.sum(train_y==1) == train_y.shape[0] or np.sum(train_y==-1) == train_y.shape[0]:
        uni = np.ones(n_km) / np.linalg.norm(np.ones(n_km))
        print "extemely biased label, setting to uni weights"
        return uni

    lambds = [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2]
    scores = []
    for lambd in lambds:
        model = pegasos.PegasosSVMClassifier(lambda_reg=lambd)
        model.fit(train_X, train_y)
        score = model.score(test_X, test_y)
        scores.append(score)

    bestlambda = lambds[np.argmax(scores)]
    model = pegasos.PegasosSVMClassifier(lambda_reg=bestlambda)
    model.fit(train_X, train_y)
    #print scores
    print ".. best lambda", bestlambda
    print "..",model.weight_vector.weights
    return model.weight_vector.weights


