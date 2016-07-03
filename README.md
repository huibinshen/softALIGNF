softALIGNF
==========

This is a python implementation of soft ALIGNF, where the solution of ALIGNF
is upper bounded. The detail can be refered to the following publication.

Dependencies:
=============

- Python >= 2.7
- Numpy >= 1.4.0
- [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/) 3.17 python interface
- [Scikit-learn](http://scikit-learn.org/stable/)
- [mosek 7](https://www.mosek.com/resources/downloads)

Instructions:
=============

The package contains two set of experiments: handle each label independently
using SVM and predict multiple labels at the same time with OVKR (Operator 
valued kernel regression).

The package contains several small datasets (no more than 2000 examples) used in
the above publication. The original large image annotation datasets are taken 
from the [paper](http://link.springer.com/article/10.1007/s11263-010-0338-6) and multi-class bioinformatics datasets are taken from the [paper](Multiclass multiple kernel learning).

Handle label indpendently using SVM (svm_code folder)
-----------------------------------------------------

- run_mkl.py. Apply different mkl methods (ALIGNF+, UNIMKL, ALIGNF, TSMKL) to
a set of datasets. The learned kernel weights are saved in svm_result folder.

- run_svm.py. Combine the kernel with the learned weights and use svm to predict
labels independently. The prediction are saved in svm_result folder.

- run_mkl_noise.py. Add noise on the labels before testing MKL methods. The 
learned kernel weights are saved in svm_result folder.

- run_svm_noise.py. Using the learned kernel weights which is based from noisy
labels to combine the kernels and testing svm performance on the uncorrupted labels. 
The prediction are saved in ovkr_result folder.

Handle labels jointly using OVKR (ovkr_code folder)
---------------------------------------------------

- run_mkl.py. Apply different mkl methods (ALIGNF+, ALIGNF) to a set of multilabel
datasets. The learned kernel weights are saved in ovkr_result folder.

- run_ovkr.py. Combine the kernel with the learned weights and use ovkr to predict
labels jointly. The prediction are saved in ovkr_result folder.

- run_mkl_noise.py. Add noise on the labels before testing MKL methods. The 
learned kernel weights are saved in ovkr_result folder.

- run_ovkr_noise.py. Using the learned kernel weights which is based from noisy
labels to combine the kernels and testing ovkr performance on the uncorrupted labels. 
The prediction are saved in ovkr_result folder.


