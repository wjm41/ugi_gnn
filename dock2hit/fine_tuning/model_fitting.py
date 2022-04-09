import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Kernel, RationalQuadratic, DotProduct, NormalizedKernelMixin, Hyperparameter, Matern

from dock2hit.fine_tuning.uncertain_forest import UncertainRandomForestRegressor


def fit_forest(X, y, uncertain=False):
    if uncertain:
        model = UncertainRandomForestRegressor()
    else:
        model = RandomForestRegressor()

    # params = {'n_estimators': [100, 1000, 10000], 'max_depth': [1, 2, 3], 'min_samples_split': [2, 4]}

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=50)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 10, num=10)]
    # max_depth = [int(x) for x in np.linspace(1, 110, num = 10)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6, 8, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4, 5]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    params = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}
    # params = {'n_estimators': [1000], 'max_depth': [1, 2]}
    search = RandomizedSearchCV(
        model, params, n_iter=1, n_jobs=-1, cv=5, scoring='neg_mean_squared_error', verbose=1)
    model = search.fit(X, y).best_estimator_
    print(search.best_params_)
    return model


def fit_gp(X, y):
    #  ARD: 0.1*np.ones(X[0].shape[1])

    # cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=None)

    # define grid
    nu_list = [0.5, 1.5, 2.5]

    kernel_list = []
    for nu in nu_list:
        kernel_list.append(1*Matern(0.1, nu=nu))
        kernel_list.append(1*Matern(0.1*np.ones(len(X[0])), nu=nu))

    kernel_list.append(1*RBF(0.1))
    kernel_list.append(1*RBF(0.1*np.ones(len(X[0]))))
    # grid['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(),  1 *
    #                 RationalQuadratic(), 1*WhiteKernel()]

    params = {'kernel': kernel_list}
    # define search
    search = GridSearchCV(GaussianProcessRegressor(alpha=1e-6,
                                                   normalize_y=True), params, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    # perform the search
    results = search.fit(X, y)
    # print('Best params_: {}'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    params = results.cv_results_['params']
    # for mean, param in zip(means, params):
    #     print("R2>%.3f with: %r" % (mean, param))
    return results.best_estimator_
