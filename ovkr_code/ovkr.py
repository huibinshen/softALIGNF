import numpy as np
from sklearn.metrics import  roc_auc_score

def normalize_km(K):
    D = np.diag(1/np.sqrt(np.diag(K)))
    return np.dot(np.dot(D,K),D)

def OVKR_train(Kx, Y, mu, lambd):
    """
    Operator-valued kernel regression for Kx(ui,uj) = kx(ui,uj)*A;
    A is chosen to have a graph-based regularizer over labels

    INPUTS
    Kx:               input gram matrix of the scalar kernel kx (size n*n)
    Y:                ouputs matrix (size d*n)
    mu:               parameter of the matrix A: value between 0 (independent) and 1
    lambd:            the regulation parameter for regression

    OUTPUTS
    AP: the learned model parameter
    """
    n_samples, n_labels = Y.shape
    M = np.dot(Y.T, Y)
    M = normalize_km(M)
    L_M = np.diag(np.sum(M,1)) - M
    A = np.linalg.pinv(mu*L_M + (1-mu)*np.diag(np.ones(n_labels)))

    d, V = np.linalg.eigh(A)

    AP = np.zeros((n_labels, n_samples))
    for i in xrange(n_labels):
        DK = lambd*np.diag(np.ones(n_samples)) + d[i]*Kx
        X = np.linalg.solve(DK.T, np.dot(Y, V[:,i])).reshape(1, n_samples)
        v = V[:,i].reshape(n_labels, 1)
        AP += d[i] * np.dot(v, X)
    return AP.T

def OVKR_test(K_test, AP):
    return np.dot(K_test, AP)

def OVKR_train_CV(Kx, Y, tags):
    n_samples, n_labels = Y.shape
    folds = np.unique(tags)
    mus = [0.1, 0.3, 0.5, 0.7, 0.9]
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    best_AUC = 0
    best_mu = 0.1
    best_lambd = 1e-4
    for i in xrange(len(mus)):
        for j in xrange(len(lambdas)):
            mu = mus[i]
            lambd = lambdas[j]
            pred = np.zeros(Y.shape)
            for fold in folds:
                validate = np.array(tags==fold)
                train = np.array(~validate)
                validate_y = Y[validate,:]
                train_y = Y[train,:]
                train_km = Kx[np.ix_(train, train)]
                validate_km = Kx[np.ix_(validate, train)]
                AP = OVKR_train(train_km, train_y, mu, lambd)
                pred[validate, :] = OVKR_test(validate_km, AP)

            AUCs = []
            for t in xrange(n_labels):
                if np.sum(Y[:,t]==1) == n_samples or np.sum(Y[:,t]==-1) == n_samples:
                    continue
                auc = roc_auc_score(Y[:, t], pred[:,t])
                AUCs.append(auc)

            AUC = np.mean(AUCs)
            if np.mean(AUC) > best_AUC:
                best_AUC = AUC
                best_mu = mu
                best_lambd = lambd
    #print "best mu:",best_mu,"best lambda",best_lambd
    return OVKR_train(Kx, Y, best_mu, best_lambd)

                

        
