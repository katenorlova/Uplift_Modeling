from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold

class Rlearner(object):
    def __init__(self, learner=None, model_w=None, model_y=None, n_fold=3, method='original', ate_alpha=.05, random_state=None):
        assert learner is not None
        self.learner = learner
        if model_w is None:
            self.model_w = deepcopy(learner)
        else:
            self.model_w = model_w
            
        if model_y is None:
            self.model_y = deepcopy(learner)
        else:
            self.model_y = model_y
            
        self.skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)    
        self.random_state = random_state
        assert method == 'original' or method == 'casualml'
        self.method = method
        self.ate_alpha = ate_alpha
        
        
    def fit(self, X, y, w, encoded_w_y):
        new_df = X.copy()
        new_df['y_predict'] = 0
        y_predict = new_df['y_predict'].copy()
        w_proba = new_df['y_predict'].copy()
        for train_ind, valid_ind in self.skf.split(X, encoded_w_y):
        
            train_id = X.iloc[train_ind].index
            valid_id = X.iloc[valid_ind].index

            self.model_y.fit(X.loc[train_id], y.loc[train_id])#predict outcomes
            pred = self.model_y.predict(X.loc[valid_id])
            y_predict.loc[valid_id] = pred

            self.model_w.fit(X.loc[train_id], w.loc[train_id])#predict outcomes
            pred_proba = self.model_w.predict_proba(X.loc[valid_id])
            w_proba.loc[valid_id] = pred_proba[:, 1]
            
        y_y_pred = y.values - y_predict.values
        w_w_prob = w.values - w_proba.values
        
        if self.method == 'original':
            # print(X.shape, np.vstack(([1]*X.shape[0], X.T)).shape)
            X_scl_tau = (w_w_prob * np.vstack(([1]*X.shape[0], X.T))).T
            # print(X_scl_tau.shape)
            self.learner.fit(X_scl_tau, y_y_pred)
            
        if self.method == 'casualml':
            self.learner.fit(X, y_y_pred/w_w_prob, sample_weight=np.square(w_w_prob))
        self.vars_c = (y[w == 0] - y_predict[w == 0]).var()
        self.vars_t = (y[w == 1] - y_predict[w == 1]).var()
    
    def predict(self, X):
        if self.method == 'original':
            return np.vstack(([1]*X.shape[0], X.T)).T@self.learner.coef_
            
        if self.method == 'casualml':
            return self.learner.predict(X)
    
    def fit_predict(self, X, y, w, encoded_w_y):
        self.fit(X, y, w, encoded_w_y)
        y_pred = self.predict(X)
        return y_pred
    
    def estimate_ate(self, X, y, w, encoded_w_y):
        y_pred = self.fit_predict(X, y, w, encoded_w_y)
        prob_w = float(sum(w)) / X.shape[0]
        
        ate = np.mean(y_pred)
        se = (np.sqrt((self.vars_t / prob_w) + (self.vars_c / (1 - prob_w)) + y_pred.var()) / X.shape[0])

        ate_lb = ate - se * norm.ppf(1 - self.ate_alpha / 2)
        ate_ub = ate + se * norm.ppf(1 - self.ate_alpha / 2)
        return ate, ate_lb, ate_ub
