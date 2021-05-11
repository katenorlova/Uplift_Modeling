from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier

class Slearner(object):
    def __init__(self, model=None, ate_alpha=.05, random_state=None):
        
        if model is None:
            self.model = GradientBoostingClassifier()
        else:
            self.model = model
            
        self.random_state = random_state
        self.ate_alpha = ate_alpha
        
        
    def fit(self, X, y, w):
        X_new = X.copy()
        X_new['w'] = w
        #pd.concat((X,w),axis=1)
        self.model.fit(X_new, y)
    
    def predict(self, X):
        X_new_0 = X.copy()
        X_new_0['w'] = pd.Series(np.zeros(X.shape[0]),index=X.index)
        y_predict_control = self.model.predict_proba(X_new_0)[:, 1]
        
        X_new_1 = X.copy()
        X_new_1['w'] = pd.Series(np.ones(X.shape[0]),index=X.index)
        y_predict_treatment = self.model.predict_proba(X_new_1)[:, 1]

        self.uplift = y_predict_treatment - y_predict_control
        
        return self.uplift
    
    def fit_predict(self, X, y, w):
        self.fit(X, y, w)
        y_pred = self.predict(X)
        return y_pred
    
    def estimate_ate(self, X, y, w):
        y_pred = self.fit_predict(X, y, w)
        self.vars_c = (y[w == 0] - y_pred[w == 0]).var()
        self.vars_t = (y[w == 1] - y_pred[w == 1]).var()
        
        prob_w = float(sum(w)) / X.shape[0]
        
        ate = np.mean(y_pred)
        se = (np.sqrt((self.vars_t / prob_w) + (self.vars_c / (1 - prob_w)) + y_pred.var()) / X.shape[0])

        ate_lb = ate - se * norm.ppf(1 - self.ate_alpha / 2)
        ate_ub = ate + se * norm.ppf(1 - self.ate_alpha / 2)
        return ate, ate_lb, ate_ub