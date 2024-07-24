# -*- coding: utf-8 -*-
import gc
import psutil
import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr


class CombinedKernelRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, gamma1=0.0001, gamma2=0.0001, weight1=0.5):
        self.alpha = alpha
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.weight1 = weight1
        self.krr = KernelRidge(alpha=alpha, kernel="precomputed")

    # Gaussian kernel function
    def _gaussian_kernel(self, X1, X2, gamma):
        pairwise_sq_dists = -2 * np.dot(X1, X2.T) + np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)
        return np.exp(-gamma * pairwise_sq_dists)

    # Model training function
    def fit(self, X, y, n_features1, n_features2):
        self.n_features1 = n_features1
        self.n_features2 = n_features2
        X1, X2 = X[:, :self.n_features1], X[:, self.n_features1:]
        K1 = self._gaussian_kernel(X1, X1, self.gamma1)
        K2 = self._gaussian_kernel(X2, X2, self.gamma2)
        K = self.weight1 * K1 + (1 - self.weight1) * K2
        self.krr.fit(K, y)
        self.X_fit_ = X
        return self   

    # Model prediction function
    def predict(self, X):
        X1, X2 = X[:, :self.n_features1], X[:, self.n_features1:]
        K1 = self._gaussian_kernel(X1, self.X_fit_[:, :self.n_features1], self.gamma1)
        K2 = self._gaussian_kernel(X2, self.X_fit_[:, self.n_features1:], self.gamma2)
        K = self.weight1 * K1 + (1 - self.weight1) * K2
        return self.krr.predict(K)
        
# Pearson correlation coefficient score
def pearson_corr_coef(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]
    
pearson_scorer = make_scorer(pearson_corr_coef, greater_is_better=True)

# print start time
print(f"Program started at: {datetime.now()}")

# load in training set data
X1_train = pd.read_csv('./example_data/train_geno.txt', delim_whitespace=True, header=None).values[:, 1:].astype(float)
X2_train = pd.read_csv('./example_data/train_TPM.txt', delim_whitespace=True, header=None).values[:, 1:].astype(float)
y_train = pd.read_csv('./example_data/train_y.txt', delim_whitespace=True, header=None).values[:, 1].astype(float)

# horizontal stacking
X_train = np.hstack([X1_train, X2_train])

# number of features
n_features1 = X1_train.shape[1]
n_features2 = X2_train.shape[1]


# load in test set data
X1_test = pd.read_csv('./example_data/test_geno.txt', delim_whitespace=True, header=None).values[:, 1:].astype(float)
X2_test = pd.read_csv('./example_data/test_TPM.txt', delim_whitespace=True, header=None).values[:, 1:].astype(float)

# horizontal stacking
X_test = np.hstack([X1_test, X2_test]) 

# release memory
del X1_train, X2_train, X1_test, X2_test
gc.collect()

print(f"Memory usage before BayesSearchCV: {psutil.virtual_memory().percent}%")

# define parameter space
param_space = {
    "alpha": Real(0.001, 1, prior="log-uniform"),
    "gamma1": Real(1e-7, 1e-3, prior="log-uniform"),
    "gamma2": Real(1e-7, 1e-3, prior="log-uniform"),
    "weight1": Real(0.1, 0.9),
}

# Bayesian optimization for hyperparameter search
opt = BayesSearchCV(CombinedKernelRidge(), param_space, n_iter=100, cv=5, n_jobs=1, random_state=42, scoring=pearson_scorer)

try:
    opt.fit(X_train, y_train, n_features1=n_features1, n_features2=n_features2)
except Exception as e:
    print(f"Exception occurred during BayesSearchCV fitting: {e}")

print(f"Memory usage after BayesSearchCV: {psutil.virtual_memory().percent}%")

# creat a folder
os.makedirs('results', exist_ok=True)

# Print the optimal hyperparameters
best_params = opt.best_params_
print("Best parameters found: ", best_params)

# save the optimal hyperparameters
with open('./results/Best_params.txt', 'w') as w:
    for k,v in best_params.items():
        w.write(str(k)+'\t'+str(v)+'\n')
    weight2 = round((1 - round(best_params['weight1'], 1)),1)
    w.write(f"weight2\t{weight2}\n")

# train the model using optimal hyperparameters
best_model = CombinedKernelRidge(**best_params)
best_model.fit(X_train, y_train, n_features1=n_features1, n_features2=n_features2)

# make predictions
y_pred = best_model.predict(X_test)

# save prediction results
with open('./example_data/test_geno.txt' ,'r') as r, open('./results/Predicted_test_y.txt', 'w') as w:              
    test_id = [i.split()[0] for i in r]

    for index, i in enumerate(y_pred):
        w.write(f"{test_id[index]}\t{i}\n")

print(f"Memory usage after predictions: {psutil.virtual_memory().percent}%")

# print end time
print(f"Program ended at: {datetime.now()}")
