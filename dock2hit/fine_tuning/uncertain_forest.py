import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

class UncertainRandomForestRegressor(RandomForestRegressor):
    """Adapted from scikit-optimize.
    https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py
    """

    def __init__(self,
                 n_estimators='warn',
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
             n_estimators=n_estimators,
             criterion=criterion,
             max_depth=max_depth,
             min_samples_split=min_samples_split,
             min_samples_leaf=min_samples_leaf,
             min_weight_fraction_leaf=min_weight_fraction_leaf,
             max_features=max_features,
             max_leaf_nodes=max_leaf_nodes,
             min_impurity_decrease=min_impurity_decrease,
             min_impurity_split=min_impurity_split,
             bootstrap=bootstrap,
             oob_score=oob_score,
             n_jobs=n_jobs,
             random_state=random_state,
             verbose=verbose,
             warm_start=warm_start)

    def get_params(self, deep=True):
        """This method overrides the one inherited from sklearn.base.BaseEstimator
        which when trying to inspect instances of this class would throw a
        RuntimeError complaining that "scikit-learn estimators should always specify
        their parameters in the signature of their __init__ (no varargs).
        Constructor (self, *args, **kwargs) doesn't  follow this convention.".
        sklearn enforces this to be able to read and set the parameter names
        in meta algorithms like pipeline and grid search which we don't need.
        """
        return self.params

    def predict(self, X_test, no_var=True, get_aleat=False, aleat_only=False):
        """Predict continuous labels y_pred and uncertainty y_var
        (unless no_var=True) for X_test.
​
        Args:
            X_test (array-like, shape=(n_samples, n_features)): Input data.
            no_var=False (bool, optional): Don't return y_var if set to true.
​
        Returns:
            2- or 1-tuple: y_pred (and y_var)
        """
        if self.criterion != "mse":
            err = f"Expected impurity to be 'mse', instead got {self.criterion}."
            raise ValueError(err)
            
        y_pred = super().predict(X_test)
        if no_var:
            return y_pred
        if aleat_only:
            y_var_aleat = self.get_var(X_test, y_pred, aleat_only)
            return y_pred, y_var_aleat
        y_var, y_var_aleat = self.get_var(X_test, y_pred, aleat_only)
        if get_aleat:
            return y_pred, y_var, y_var_aleat
        return y_pred, y_var

    def get_var(self, X_test, y_pred, aleat_only=False):
        """Computes var(Y|X_test) via law of total variance E[Var(Y|Tree)] + Var(E[Y|Tree]).
        Note: Another option for estimating confidence intervals is be prediction
        variability, i.e. how influential training set is for producing observed
        random forest predictions. Implemented in
        https://github.com/scikit-learn-contrib/forest-confidence-interval.
        However, empirically our method of obtaining y_var seems to be more accurate.
​
        Args:
            X_test (array-like, shape=(n_samples, n_features)): Input data.
            y_pred (array-like, shape=(n_samples,)): Prediction for each sample
                as returned by RFR.predict(X_test).
​
        Returns:
            array-like, shape=(n_samples,): Standard deviation of y_pred at X_test.
            Since self.criterion is set to "mse", var[i] ~= var(y | X_test[i]).
        """
        # trees is a list of fitted binary decision trees.
        y_var, y_var_aleat, trees = *np.zeros([2, len(X_test)]), self.estimators_

        # Compute var(y|X_test) as described in sec. 4.3.2
        # of http://arxiv.org/pdf/1211.0906v2.pdf.
        for tree in trees:
            # Tree impurity modelling aleatoric uncertainty.
            var_tree = tree.tree_.impurity[tree.apply(X_test)]
            y_pred_tree = tree.predict(X_test)
            y_var += var_tree + y_pred_tree ** 2
            y_var_aleat += var_tree

        y_var /= len(trees)
        y_var_aleat /= len(trees)
        y_var -= y_pred ** 2
        if aleat_only:
            return y_var_aleat
        return y_var, y_var_aleat

    def get_corr(self, X_test, with_cov=False, alpha=1e-14):
        """Compute the Pearson correlation coefficient matrix of predictions. Each
        entry in the correlation matrix is the covariance between those random variables
        normalized by division with their respective standard deviations. See
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html.
        Args:
            X_test (array, shape=(n_samples, n_features)): Input data.

        Returns:
            array, shape=(n_samples, n_samples): Correlation coefficients between
            samples in X_test.
        """
        # Each row in preds represents a variable, in this case different samples
        # in X_test, while the columns contain a series of observations corresponding
        # to predictions from different trees in the forest.
        preds = np.array([tree.predict(X_test) for tree in self.estimators_]).T

        # Ensure the correlation matrix is positive definite despite rounding errors.
        psd = np.eye(len(X_test)) * alpha

        if with_cov:
            return np.corrcoef(preds) + psd, np.cov(preds) + psd
        return np.corrcoef(preds) + psd