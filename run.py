from webapp import app
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import linear_model
import scipy.stats as stat
import numpy as np

if __name__ == "__main__":
    class LogisticRegression_with_p_values:
        
        def __init__(self,*args,**kwargs):#,**kwargs):
            self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

        def fit(self,X,y):
            self.model.fit(X,y)
            denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
            denom = np.tile(denom,(X.shape[1],1)).T
            F_ij = np.dot((X / denom).T,X)
            Cramer_Rao = np.linalg.inv(F_ij)
            sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
            z_scores = self.model.coef_[0] / sigma_estimates
            p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
            
            self.coef_ = self.model.coef_
            self.intercept_ = self.model.intercept_
            self.p_values = p_values

    app.run(debug=True)
