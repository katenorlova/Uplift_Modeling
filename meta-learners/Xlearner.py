from copy import deepcopy
import numpy as np
# import pandas as pd


class Xlearner(object):
    def __init__(self, learner=None, control_outcome_learner=None, treatment_outcome_learner=None,
                 control_effect_learner=None, treatment_effect_learner=None, ate_alpha=0.05, propensity=0.9):
        assert (learner is not None) or ((control_outcome_learner is not None) and
                                         (treatment_outcome_learner is not None) and
                                         (control_effect_learner is not None) and
                                         (treatment_effect_learner is not None))

        if control_outcome_learner is None:
            self.model_mu_c = deepcopy(learner)
        else:
            self.model_mu_c = control_outcome_learner

        if treatment_outcome_learner is None:
            self.model_mu_t = deepcopy(learner)
        else:
            self.model_mu_t = treatment_outcome_learner

        if control_effect_learner is None:
            self.model_tau_c = deepcopy(learner)
        else:
            self.model_tau_c = control_effect_learner

        if treatment_effect_learner is None:
            self.model_tau_t = deepcopy(learner)
        else:
            self.model_tau_t = treatment_effect_learner

        self.ate_alpha = ate_alpha
        self.propensity = propensity

    def fit(self, X_in, w_in, y_in):
        X = np.asarray(X_in, dtype=np.float32)
        w = np.asarray(w_in, dtype=np.float32)
        y = np.asarray(y_in, dtype=np.float32)
        d = np.zeros_like(y, dtype=np.float32)

        mask_t = w == 1
        mask_c = w == 0
        X_t = X[mask_t]
        X_c = X[mask_c]
        w_t = w[mask_t]
        w_c = w[mask_c]
        y_t = y[mask_t]
        y_c = y[mask_c]

        self.model_mu_c.fit(X_c, y_c)  # учимся предсказывать ауткам на контроле
        self.model_mu_t.fit(X_t, y_t)  # учимся предсказывать ауткам на тритменте

        mu0_x1 = self.model_mu_c.predict(X_t)
        mu1_x0 = self.model_mu_t.predict(X_c)
        d[mask_t] = y_t - mu0_x1
        d[mask_c] = mu1_x0 - y_c

        self.model_tau_t.fit(X_t, d[mask_t])
        self.model_tau_c.fit(X_c, d[mask_c])


    def predict(self, X_in):
        X = np.asarray(X_in)
        g = self.propensity
        pred = g * self.model_tau_c.predict(X) + (1 - g) * self.model_tau_t.predict(X)

        return pred

    def fit_predict(self, X_in, w_in, y_in):
        X, w, y = deepcopy(X_in), deepcopy(w_in), deepcopy(y_in)
        self.fit(X, w, y)
        pred = self.predict(X)

        return pred