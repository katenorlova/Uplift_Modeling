from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class Tlearner(object):
    def __init__(self, model1=None, model2=None, ate_alpha=.05, random_state=None):
        
        if model1 is None:
            self.model1 = LogisticRegression()
        else:
            self.model1 = model1
            
        if model2 is None:
            self.model2 =  RandomForestClassifier(n_estimators=100, max_depth=5)
        else:
            self.model2 = model2   
            
        self.random_state = random_state
        self.ate_alpha = ate_alpha
        
        
    def fit(self, X, y, w):
        X_treatment, y_treatment = X[w==1],y[w==1]
        X_control, y_control = X[w==0],y[w==0]

        self.model1.fit(X_treatment, y_treatment)
        self.model2.fit(X_control,y_control)
    
    def predict(self, X):
        
        y_predict_treatment = self.model1.predict_proba(X)[:, 1]
        y_predict_control = self.model2.predict_proba(X)[:, 1]

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
