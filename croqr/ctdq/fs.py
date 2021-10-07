# mlextend
# import datetime
import types
from copy import deepcopy
from itertools import combinations

import scipy as sp
import scipy.stats
import tensorflow as tf
from ct.utils import *
from joblib import Parallel, delayed
from pdpbox import pdp, info_plots
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.multiclass import type_of_target
from tensorflow import keras

pd.set_option('display.width', None)


def scalercheck_output(X, ensure_index=None, ensure_columns=None):
    if ensure_index is not None:
        if ensure_columns is not None:
            if type(ensure_index) is pd.DataFrame and type(ensure_columns) is pd.DataFrame:
                X = pd.DataFrame(X, index=ensure_index.index, columns=ensure_columns.columns)
        else:
            if type(ensure_index) is pd.DataFrame:
                X = pd.DataFrame(X, index=ensure_index.index)
    return X


from sklearn.preprocessing import StandardScaler as _StandardScaler


class StandardScaler(_StandardScaler):
    def transform(self, X):
        Xt = super(StandardScaler, self).transform(X)
        return scalercheck_output(Xt, ensure_index=X, ensure_columns=X)


class _BaseXComposition(_BaseComposition):
    """
    parameter handler for list of estimators
    """

    def _set_params(self, attr, named_attr, **params):
        # Ordered parameter replacement
        # 1. root parameter
        if attr in params:
            setattr(self, attr, params.pop(attr))

        # 2. single estimator replacement
        items = getattr(self, named_attr)
        names = []
        if items:
            names, estimators = zip(*items)
            estimators = list(estimators)
        for name in list(six.iterkeys(params)):
            if '__' not in name and name in names:
                # replace single estimator and re-build the
                # root estimators list
                for i, est_name in enumerate(names):
                    if est_name == name:
                        new_val = params.pop(name)
                        if new_val is None:
                            del estimators[i]
                        else:
                            estimators[i] = new_val
                        break
                # replace the root estimators
                setattr(self, attr, estimators)

        # 3. estimator parameters and other initialisation arguments
        super(_BaseXComposition, self).set_params(**params)
        return self


def _calc_score(selector, X, y, indices, groups=None, **fit_params):
    if selector.cv:
        scores = cross_val_score(selector.est_,
                                 X[:, indices], y,
                                 groups=groups,
                                 cv=selector.cv,
                                 scoring=selector.scorer,
                                 n_jobs=1,
                                 pre_dispatch=selector.pre_dispatch,
                                 fit_params=fit_params)
    else:
        selector.est_.fit(X[:, indices], y, **fit_params)
        scores = np.array([selector.scorer(selector.est_, X[:, indices], y)])
    return indices, scores


def _get_featurenames(subsets_dict, feature_idx, custom_feature_names, X):
    feature_names = None
    if feature_idx is not None:
        if custom_feature_names is not None:
            feature_names = tuple((custom_feature_names[i]
                                   for i in feature_idx))
        elif hasattr(X, 'loc'):
            feature_names = tuple((X.columns[i] for i in feature_idx))
        else:
            feature_names = tuple(str(i) for i in feature_idx)

    subsets_dict_ = deepcopy(subsets_dict)
    for key in subsets_dict_:
        if custom_feature_names is not None:
            new_tuple = tuple((custom_feature_names[i]
                               for i in subsets_dict[key]['feature_idx']))
        elif hasattr(X, 'loc'):
            new_tuple = tuple((X.columns[i]
                               for i in subsets_dict[key]['feature_idx']))
        else:
            new_tuple = tuple(str(i) for i in subsets_dict[key]['feature_idx'])
        subsets_dict_[key]['feature_names'] = new_tuple

    return subsets_dict_, feature_names


class SequentialFeatureSelector(_BaseXComposition, MetaEstimatorMixin):
    """Sequential Feature Selection for Classification and Regression.
    Parameters
    ----------
    estimator : scikit-learn classifier or regressor
    k_features : int or tuple or str (default: 1)
        Number of features to select,
        where k_features < the full feature set.
        New in 0.4.2: A tuple containing a min and max value can be provided,
            and the SFS will consider return any feature combination between
            min and max that scored highest in cross-validtion. For example,
            the tuple (1, 4) will return any combination from
            1 up to 4 features instead of a fixed number of features k.
        New in 0.8.0: A string argument "best" or "parsimonious".
            If "best" is provided, the feature selector will return the
            feature subset with the best cross-validation performance.
            If "parsimonious" is provided as an argument, the smallest
            feature subset that is within one standard error of the
            cross-validation performance will be selected.
    forward : bool (default: True)
        Forward selection if True,
        backward selection otherwise
    floating : bool (default: False)
        Adds a conditional exclusion/inclusion if True.
    verbose : int (default: 0), level of verbosity to use in logging.
        If 0, no output,
        if 1 number of features in current set, if 2 detailed logging i
        ncluding timestamp and cv scores at step.
    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.
    cv : int (default: 5)
        Integer or iterable yielding train, test splits. If cv is an integer
        and `estimator` is a classifier (or y consists of integer class
        labels) stratified k-fold. Otherwise regular k-fold cross-validation
        is performed. No cross-validation if cv is None, False, or 0.
    n_jobs : int (default: 1)
        The number of CPUs to use for evaluating different feature subsets
        in parallel. -1 means 'all CPUs'.
    pre_dispatch : int, or string (default: '2*n_jobs')
        Controls the number of jobs that get dispatched
        during parallel execution if `n_jobs > 1` or `n_jobs=-1`.
        Reducing this number can be useful to avoid an explosion of
        memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
        None, in which case all the jobs are immediately created and spawned.
            Use this for lightweight and fast-running jobs,
            to avoid delays due to on-demand spawning of the jobs
        An int, giving the exact number of total jobs that are spawned
        A string, giving an expression as a function
            of n_jobs, as in `2*n_jobs`
    clone_estimator : bool (default: True)
        Clones estimator if True; works with the original estimator instance
        if False. Set to False if the estimator doesn't
        implement scikit-learn's set_params and get_params methods.
        In addition, it is required to set cv=0, and n_jobs=1.
    fixed_features : tuple (default: None)
        If not `None`, the feature indices provided as a tuple will be
        regarded as fixed by the feature selector. For example, if
        `fixed_features=(1, 3, 7)`, the 2nd, 4th, and 8th feature are
        guaranteed to be present in the solution. Note that if
        `fixed_features` is not `None`, make sure that the number of
        features to be selected is greater than `len(fixed_features)`.
        In other words, ensure that `k_features > len(fixed_features)`.
        New in mlxtend v. 0.18.0.
    Attributes
    ----------
    k_feature_idx_ : array-like, shape = [n_predictions]
        Feature Indices of the selected feature subsets.
    k_feature_names_ : array-like, shape = [n_predictions]
        Feature names of the selected feature subsets. If pandas
        DataFrames are used in the `fit` method, the feature
        names correspond to the column names. Otherwise, the
        feature names are string representation of the feature
        array indices. New in v 0.13.0.
    k_score_ : float
        Cross validation average score of the selected subset.
    subsets_ : dict
        A dictionary of selected feature subsets during the
        sequential selection, where the dictionary keys are
        the lengths k of these feature subsets. The dictionary
        values are dictionaries themselves with the following
        keys: 'feature_idx' (tuple of indices of the feature subset)
              'feature_names' (tuple of feature names of the feat. subset)
              'cv_scores' (list individual cross-validation scores)
              'avg_score' (average cross-validation score)
        Note that if pandas
        DataFrames are used in the `fit` method, the 'feature_names'
        correspond to the column names. Otherwise, the
        feature names are string representation of the feature
        array indices. The 'feature_names' is new in v 0.13.0.
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
    """

    def __init__(self, estimator, k_features=1,
                 forward=True, floating=False,
                 verbose=0, scoring=None,
                 cv=5, n_jobs=1,
                 pre_dispatch='2*n_jobs',
                 clone_estimator=True,
                 fixed_features=None):

        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.pre_dispatch = pre_dispatch
        # Want to raise meaningful error message if a
        # cross-validation generator is inputted
        if isinstance(cv, types.GeneratorType):
            err_msg = ('Input cv is a generator object, which is not '
                       'supported. Instead please input an iterable yielding '
                       'train, test splits. This can usually be done by '
                       'passing a cross-validation generator to the '
                       'built-in list function. I.e. cv=list(<cv-generator>)')
            raise TypeError(err_msg)
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.clone_estimator = clone_estimator

        if fixed_features is not None:
            if isinstance(self.k_features, int) and \
                    self.k_features <= len(fixed_features):
                raise ValueError('Number of features to be selected must'
                                 ' be larger than the number of'
                                 ' features specified via `fixed_features`.'
                                 ' Got `k_features=%d` and'
                                 ' `fixed_features=%d`' %
                                 (k_features, len(fixed_features)))

            elif isinstance(self.k_features, tuple) and \
                    self.k_features[0] <= len(fixed_features):
                raise ValueError('The minimum number of features to'
                                 ' be selected must'
                                 ' be larger than the number of'
                                 ' features specified via `fixed_features`.'
                                 ' Got `k_features=%s` and '
                                 '`len(fixed_features)=%d`' %
                                 (k_features, len(fixed_features)))

        self.fixed_features = fixed_features

        if self.clone_estimator:
            self.est_ = clone(self.estimator)
        else:
            self.est_ = self.estimator
        self.scoring = scoring

        if scoring is None:
            if self.est_._estimator_type == 'classifier':
                scoring = 'accuracy'
            elif self.est_._estimator_type == 'regressor':
                scoring = 'r2'
            else:
                raise AttributeError('Estimator must '
                                     'be a Classifier or Regressor.')
        if isinstance(scoring, str):
            self.scorer = get_scorer(scoring)
        else:
            self.scorer = scoring

        self.fitted = False
        self.subsets_ = {}
        self.interrupted_ = False

        # don't mess with this unless testing
        self._TESTING_INTERRUPT_MODE = False

    def get_params(self, deep=True):
        #
        # Return estimator parameter names for GridSearch support.
        #
        return self._get_params('named_estimators', deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params('estimator', 'named_estimators', **params)
        return self

    def fit(self, X, y, custom_feature_names=None, groups=None, **fit_params):
        """Perform feature selection and learn model from training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y : array-like, shape = [n_samples]
            Target values.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for y.
        custom_feature_names : None or tuple (default: tuple)
            Custom feature names for `self.k_feature_names` and
            `self.subsets_[i]['feature_names']`.
            (new in v 0.13.0)
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fit_params : dict of string -> object, optional
            Parameters to pass to to the fit method of classifier.
        Returns
        -------
        self : object
        """

        # reset from a potential previous fit run
        self.subsets_ = {}
        self.fitted = False
        self.interrupted_ = False
        self.k_feature_idx_ = None
        self.k_feature_names_ = None
        self.k_score_ = None

        self.fixed_features_ = self.fixed_features
        self.fixed_features_set_ = set()

        if hasattr(X, 'loc'):
            X_ = X.values
            if self.fixed_features is not None:
                self.fixed_features_ = tuple(X.columns.get_loc(c)
                                             if isinstance(c, str) else c
                                             for c in self.fixed_features
                                             )
        else:
            X_ = X

        if self.fixed_features is not None:
            self.fixed_features_set_ = set(self.fixed_features_)

        if (custom_feature_names is not None
                and len(custom_feature_names) != X.shape[1]):
            raise ValueError('If custom_feature_names is not None, '
                             'the number of elements in custom_feature_names '
                             'must equal the number of columns in X.')

        if not isinstance(self.k_features, int) and \
                not isinstance(self.k_features, tuple) \
                and not isinstance(self.k_features, str):
            raise AttributeError('k_features must be a positive integer'
                                 ', tuple, or string')

        if (isinstance(self.k_features, int) and (
                self.k_features < 1 or self.k_features > X_.shape[1])):
            raise AttributeError('k_features must be a positive integer'
                                 ' between 1 and X.shape[1], got %s'
                                 % (self.k_features,))

        if isinstance(self.k_features, tuple):
            if len(self.k_features) != 2:
                raise AttributeError('k_features tuple must consist of 2'
                                     ' elements a min and a max value.')

            if self.k_features[0] not in range(1, X_.shape[1] + 1):
                raise AttributeError('k_features tuple min value must be in'
                                     ' range(1, X.shape[1]+1).')

            if self.k_features[1] not in range(1, X_.shape[1] + 1):
                raise AttributeError('k_features tuple max value must be in'
                                     ' range(1, X.shape[1]+1).')

            if self.k_features[0] > self.k_features[1]:
                raise AttributeError('The min k_features value must be smaller'
                                     ' than the max k_features value.')

        if isinstance(self.k_features, tuple) or \
                isinstance(self.k_features, str):

            select_in_range = True

            if isinstance(self.k_features, str):
                if self.k_features not in {'best', 'parsimonious'}:
                    raise AttributeError('If a string argument is provided, '
                                         'it must be "best" or "parsimonious"')
                else:
                    min_k = 1
                    max_k = X_.shape[1]
            else:
                min_k = self.k_features[0]
                max_k = self.k_features[1]

        else:
            select_in_range = False
            k_to_select = self.k_features

        orig_set = set(range(X_.shape[1]))
        n_features = X_.shape[1]

        if self.forward and self.fixed_features is not None:
            orig_set = set(range(X_.shape[1])) - self.fixed_features_set_
            n_features = len(orig_set)

        if self.forward:
            if select_in_range:
                k_to_select = max_k

            if self.fixed_features is not None:
                k_idx = self.fixed_features_
                k = len(k_idx)
                k_idx, k_score = _calc_score(self, X_, y, k_idx,
                                             groups=groups, **fit_params)
                self.subsets_[k] = {
                    'feature_idx': k_idx,
                    'cv_scores': np.around(k_score, 3),
                    'avg_score': np.round(np.nanmean(k_score), 3)
                }

            else:
                k_idx = ()
                k = 0
        else:
            if select_in_range:
                k_to_select = min_k
            k_idx = tuple(orig_set)
            k = len(k_idx)
            k_idx, k_score = _calc_score(self, X_, y, k_idx,
                                         groups=groups, **fit_params)
            self.subsets_[k] = {
                'feature_idx': k_idx,
                'cv_scores': np.round(k_score, 3),
                'avg_score': np.round(np.nanmean(k_score), 3)
            }
        best_subset = None
        k_score = 0

        try:
            while k != k_to_select:
                prev_subset = set(k_idx)

                if self.forward:
                    k_idx, k_score, cv_scores = self._inclusion(
                        orig_set=orig_set,
                        subset=prev_subset,
                        X=X_,
                        y=y,
                        groups=groups,
                        **fit_params
                    )
                else:
                    k_idx, k_score, cv_scores = self._exclusion(
                        feature_set=prev_subset,
                        X=X_,
                        y=y,
                        groups=groups,
                        fixed_feature=self.fixed_features_set_,
                        **fit_params
                    )

                if self.floating:

                    if self.forward:
                        continuation_cond_1 = len(k_idx)
                    else:
                        continuation_cond_1 = n_features - len(k_idx)

                    continuation_cond_2 = True
                    ran_step_1 = True
                    new_feature = None

                    while continuation_cond_1 >= 2 and continuation_cond_2:
                        k_score_c = None

                        if ran_step_1:
                            (new_feature,) = set(k_idx) ^ prev_subset

                        if self.forward:

                            fixed_features_ok = True
                            if self.fixed_features is not None and \
                                    len(self.fixed_features) - len(k_idx) <= 1:
                                fixed_features_ok = False
                            if fixed_features_ok:
                                k_idx_c, k_score_c, cv_scores_c = \
                                    self._exclusion(
                                        feature_set=k_idx,
                                        fixed_feature=(
                                                {new_feature} |
                                                self.fixed_features_set_),
                                        X=X_,
                                        y=y,
                                        groups=groups,
                                        **fit_params
                                    )

                        else:
                            k_idx_c, k_score_c, cv_scores_c = self._inclusion(
                                orig_set=orig_set - {new_feature},
                                subset=set(k_idx),
                                X=X_,
                                y=y,
                                groups=groups,
                                **fit_params
                            )

                        if k_score_c is not None and k_score_c > k_score:

                            if len(k_idx_c) in self.subsets_:
                                cached_score = self.subsets_[len(
                                    k_idx_c)]['avg_score']
                            else:
                                cached_score = None

                            if cached_score is None or \
                                    k_score_c > cached_score:
                                prev_subset = set(k_idx)
                                k_idx, k_score, cv_scores = \
                                    k_idx_c, k_score_c, cv_scores_c
                                continuation_cond_1 = len(k_idx)
                                ran_step_1 = False

                            else:
                                continuation_cond_2 = False

                        else:
                            continuation_cond_2 = False

                k = len(k_idx)
                # floating can lead to multiple same-sized subsets
                if k not in self.subsets_ or (k_score >
                                              self.subsets_[k]['avg_score']):
                    k_idx = tuple(sorted(k_idx))
                    self.subsets_[k] = {
                        'feature_idx': k_idx,
                        'cv_scores': cv_scores,
                        'avg_score': k_score
                    }

                if self.verbose == 1:
                    sys.stderr.write('\rFeatures: %d/%s' % (len(k_idx), k_to_select))
                    sys.stderr.flush()
                elif self.verbose > 1:
                    print([X.columns[i] for i in k_idx], k_score, cv_scores)
                    sys.stderr.write('\n[%s] Features: %d/%s -- score: %s' % (
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        len(k_idx),
                        k_to_select,
                        k_score
                    ))
                logging.info(
                    f'sfs: score={np.round(k_score, 3)} cvscores={np.round(cv_scores, 3)} nf={len(k_idx)} {[X.columns[i] for i in k_idx]} ')
                if self._TESTING_INTERRUPT_MODE:
                    self.subsets_, self.k_feature_names_ = \
                        _get_featurenames(self.subsets_,
                                          self.k_feature_idx_,
                                          custom_feature_names,
                                          X)
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            self.interrupted_ = True
            sys.stderr.write('\nSTOPPING EARLY DUE TO KEYBOARD INTERRUPT...')

        if select_in_range:
            max_score = float('-inf')

            max_score = float('-inf')
            for k in self.subsets_:
                if k < min_k or k > max_k:
                    continue
                if self.subsets_[k]['avg_score'] > max_score:
                    max_score = self.subsets_[k]['avg_score']
                    best_subset = k
            k_score = max_score
            k_idx = self.subsets_[best_subset]['feature_idx']

            if self.k_features == 'parsimonious':
                for k in self.subsets_:
                    if k >= best_subset:
                        continue
                    if self.subsets_[k]['avg_score'] >= (
                            max_score - np.std(self.subsets_[k]['cv_scores']) /
                            self.subsets_[k]['cv_scores'].shape[0]):
                        max_score = self.subsets_[k]['avg_score']
                        best_subset = k
                k_score = max_score
                k_idx = self.subsets_[best_subset]['feature_idx']

        self.k_feature_idx_ = k_idx
        self.k_score_ = k_score
        self.fitted = True
        self.subsets_, self.k_feature_names_ = \
            _get_featurenames(self.subsets_,
                              self.k_feature_idx_,
                              custom_feature_names,
                              X)
        return self

    def _inclusion(self, orig_set, subset, X, y, ignore_feature=None,
                   groups=None, **fit_params):
        all_avg_scores = []
        all_cv_scores = []
        all_subsets = []
        res = (None, None, None)
        remaining = orig_set - subset
        if remaining:
            features = len(remaining)
            n_jobs = min(self.n_jobs, features)
            parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                                pre_dispatch=self.pre_dispatch)
            work = parallel(delayed(_calc_score)
                            (self, X, y,
                             tuple(subset | {feature}),
                             groups=groups, **fit_params)
                            for feature in remaining
                            if feature != ignore_feature)

            for new_subset, cv_scores in work:
                all_avg_scores.append(np.nanmean(cv_scores))
                all_cv_scores.append(cv_scores)
                all_subsets.append(new_subset)

            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])
        return res

    def _exclusion(self, feature_set, X, y, fixed_feature=None,
                   groups=None, **fit_params):
        n = len(feature_set)
        res = (None, None, None)
        if n > 1:
            all_avg_scores = []
            all_cv_scores = []
            all_subsets = []
            features = n
            n_jobs = min(self.n_jobs, features)
            parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                                pre_dispatch=self.pre_dispatch)
            work = parallel(delayed(_calc_score)(self, X, y, p,
                                                 groups=groups, **fit_params)
                            for p in combinations(feature_set, r=n - 1)
                            if not fixed_feature or
                            fixed_feature.issubset(set(p)))

            for p, cv_scores in work:
                all_avg_scores.append(np.nanmean(cv_scores))
                all_cv_scores.append(cv_scores)
                all_subsets.append(p)

            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])
        return res

    def transform(self, X):
        """Reduce X to its most important features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        Returns
        -------
        Reduced feature subset of X, shape={n_samples, k_features}
        """
        self._check_fitted()
        if hasattr(X, 'loc'):
            X_ = X.values
        else:
            X_ = X
        return X_[:, self.k_feature_idx_]

    def fit_transform(self, X, y, groups=None, **fit_params):
        """Fit to training data then reduce X to its most important features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            New in v 0.13.0: pandas DataFrames are now also accepted as
            argument for X.
        y : array-like, shape = [n_samples]
            Target values.
            New in v 0.13.0: a pandas Series are now also accepted as
            argument for y.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Passed to the fit method of the cross-validator.
        fit_params : dict of string -> object, optional
            Parameters to pass to to the fit method of classifier.
        Returns
        -------
        Reduced feature subset of X, shape={n_samples, k_features}
        """
        self.fit(X, y, groups=groups, **fit_params)
        return self.transform(X)

    def get_metric_dict(self, confidence_interval=0.95):
        """Return metric dictionary
        Parameters
        ----------
        confidence_interval : float (default: 0.95)
            A positive float between 0.0 and 1.0 to compute the confidence
            interval bounds of the CV score averages.
        Returns
        ----------
        Dictionary with items where each dictionary value is a list
        with the number of iterations (number of feature subsets) as
        its length. The dictionary keys corresponding to these lists
        are as follows:
            'feature_idx': tuple of the indices of the feature subset
            'cv_scores': list with individual CV scores
            'avg_score': of CV average scores
            'std_dev': standard deviation of the CV score average
            'std_err': standard error of the CV score average
            'ci_bound': confidence interval bound of the CV score average
        """
        self._check_fitted()
        fdict = deepcopy(self.subsets_)
        for k in fdict:
            std_dev = np.around(np.std(self.subsets_[k]['cv_scores']), decimals=3)
            bound, std_err = self._calc_confidence(
                self.subsets_[k]['cv_scores'],
                confidence=confidence_interval)
            fdict[k]['ci_bound'] = bound
            fdict[k]['std_dev'] = std_dev
            fdict[k]['std_err'] = std_err
        return fdict

    def _calc_confidence(self, ary, confidence=0.95):
        std_err = scipy.stats.sem(ary)
        bound = std_err * sp.stats.t._ppf((1 + confidence) / 2.0, len(ary))
        return bound, std_err

    def _check_fitted(self):
        if not self.fitted:
            raise AttributeError('SequentialFeatureSelector has not been'
                                 ' fitted, yet.')


from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator


class BorutaPy(BaseEstimator, TransformerMixin):
    """
    Improved Python implementation of the Boruta R package.
    The improvements of this implementation include:
    - Faster run times:
        Thanks to scikit-learn's fast implementation of the ensemble methods.
    - Scikit-learn like interface:
        Use BorutaPy just like any other scikit learner: fit, fit_transform and
        transform are all implemented in a similar fashion.
    - Modularity:
        Any ensemble method could be used: random forest, extra trees
        classifier, even gradient boosted trees.
    - Two step correction:
        The original Boruta code corrects for multiple testing in an overly
        conservative way. In this implementation, the Benjamini Hochberg FDR is
        used to correct in each iteration across active features. This means
        only those features are included in the correction which are still in
        the selection process. Following this, each that passed goes through a
        regular Bonferroni correction to check for the repeated testing over
        the iterations.
    - Percentile:
        Instead of using the max values of the shadow features the user can
        specify which percentile to use. This gives a finer control over this
        crucial parameter. For more info, please read about the perc parameter.
    - Automatic tree number:
        Setting the n_estimator to 'auto' will calculate the number of trees
        in each itartion based on the number of features under investigation.
        This way more trees are used when the training data has many feautres
        and less when most of the features have been rejected.
    - Ranking of features:
        After fitting BorutaPy it provides the user with ranking of features.
        Confirmed ones are 1, Tentatives are 2, and the rejected are ranked
        starting from 3, based on their feautre importance history through
        the iterations.
    We highly recommend using pruned trees with a depth between 3-7.
    For more, see the docs of these functions, and the examples below.
    Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/
    Boruta is an all relevant feature selection method, while most other are
    minimal optimal; this means it tries to find all features carrying
    information usable for prediction, rather than finding a possibly compact
    subset of features on which some classifier has a minimal error.
    Why bother with all relevant feature selection?
    When you try to understand the phenomenon that made your data, you should
    care about all factors that contribute to it, not just the bluntest signs
    of it in context of your methodology (yes, minimal optimal set of features
    by definition depends on your classifier choice).
    Parameters
    ----------
    estimator : object
        A supervised learning estimator, with a 'fit' method that returns the
        feature_importances_ attribute. Important features must correspond to
        high absolute values in the feature_importances_.
    n_estimators : int or string, default = 1000
        If int sets the number of estimators in the chosen ensemble method.
        If 'auto' this is determined automatically based on the size of the
        dataset. The other parameters of the used estimators need to be set
        with initialisation.
    perc : int, default = 100
        Instead of the max we use the percentile defined by the user, to pick
        our threshold for comparison between shadow and real features. The max
        tend to be too stringent. This provides a finer control over this. The
        lower perc is the more false positives will be picked as relevant but
        also the less relevant features will be left out. The usual trade-off.
        The default is essentially the vanilla Boruta corresponding to the max.
    alpha : float, default = 0.05
        Level at which the corrected p-values will get rejected in both
        correction steps.
    two_step : Boolean, default = True
        If you want to use the original implementation of Boruta with Bonferroni
        correction only set this to False.
    max_iter : int, default = 100
        The number of maximum iterations to perform.
    random_state : int, RandomState instance or None; default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays iteration number
        - 2: which features have been selected already
    Attributes
    ----------
    n_features_ : int
        The number of selected features.
    support_ : array of shape [n_features]
        The mask of selected features - only confirmed ones are True.
    support_weak_ : array of shape [n_features]
        The mask of selected tentative features, which haven't gained enough
        support during the max_iter number of iterations..
    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1 and tentative features are assigned
        rank 2.
    Examples
    --------
    
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy
    
    # load X and y
    # NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    X = pd.read_csv('examples/test_X.csv', index_col=0).values
    y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    y = y.ravel()
    
    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    
    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
    
    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)
    
    # check selected features - first 5 features are selected
    feat_selector.support_
    
    # check ranking of features
    feat_selector.ranking_
    
    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    References
    ----------
    [1] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package"
        Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010
    """

    def __init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05,
                 two_step=True, max_iter=100, random_state=None, verbose=0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fits the Boruta feature selection with the provided estimator.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """

        return self._fit(X, y)

    def transform(self, X, weak=False, return_df=False):
        """
        Reduces the input X to the features selected by Boruta.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.
        
        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        return self._transform(X, weak, return_df)

    def fit_transform(self, X, y, weak=False, return_df=False):
        """
        Fits Boruta, then reduces the input X to the selected features.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        weak: boolean, default = False
            If set to true, the tentative features are also used to reduce X.
        return_df : boolean, default = False
            If ``X`` if a pandas dataframe and this parameter is set to True,
            the transformed data will also be a dataframe.
        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features which were
            selected by Boruta.
        """

        self._fit(X, y)
        return self._transform(X, weak, return_df)

    def _validate_pandas_input(self, arg):
        try:
            return arg.values
        except AttributeError:
            raise ValueError(
                "input needs to be a numpy array or pandas data frame."
            )

    def _fit(self, X, y):
        # check input params
        self._check_params(X, y)

        if not isinstance(X, np.ndarray):
            X = self._validate_pandas_input(X)
        if not isinstance(y, np.ndarray):
            y = self._validate_pandas_input(y)

        self.random_state = check_random_state(self.random_state)
        # setup variables for Boruta
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        hit_reg = np.zeros(n_feat, dtype=np.int)
        # these record the history of the iterations
        imp_history = np.zeros(n_feat, dtype=np.float)
        sha_max_history = []

        # set n_estimators
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        while np.any(dec_reg == 0) and _iter < self.max_iter:
            # find optimal number of trees and depth
            if self.n_estimators == 'auto':
                # number of features that aren't rejected
                not_rejected = np.where(dec_reg >= 0)[0].shape[0]
                n_tree = self._get_tree_num(not_rejected)
                self.estimator.set_params(n_estimators=n_tree)

            # make sure we start with a new tree in each iteration
            self.estimator.set_params(random_state=self.random_state)

            # add shadow attributes, shuffle them and train estimator, get imps
            cur_imp = self._add_shadows_get_imps(X, y, dec_reg)

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(cur_imp[1], self.perc)

            # record importance history
            sha_max_history.append(imp_sha_max)
            imp_history = np.vstack((imp_history, cur_imp[0]))

            # register which feature is more imp than the max of shadows
            hit_reg = self._assign_hits(hit_reg, cur_imp, imp_sha_max)

            # based on hit_reg we check if a feature is doing better than
            # expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, _iter)

            # print out confirmed features
            if self.verbose > 0 and _iter < self.max_iter:
                self._print_results(dec_reg, _iter, 0)
            if _iter < self.max_iter:
                _iter += 1

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]
        # ignore the first row of zeros
        tentative_median = np.median(imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median
                                       > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=np.bool)
        self.support_weak_[tentative] = 1

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[1:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
            # calculate ranks in each iteration, then median of ranks across feats
            iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
            rank_medians = np.nanmedian(iter_ranks, axis=0)
            ranks = self._nanrankdata(rank_medians, axis=0)

            # set smallest rank to 3 if there are tentative feats
            if tentative.shape[0] > 0:
                ranks = ranks - np.min(ranks) + 3
            else:
                # and 2 otherwise
                ranks = ranks - np.min(ranks) + 2
            self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=np.bool)

        # notify user
        if self.verbose > 0:
            self._print_results(dec_reg, _iter, 1)
        return self

    def _transform(self, X, weak=False, return_df=False):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')

        if weak:
            indices = self.support_ + self.support_weak_
        else:
            indices = self.support_

        if return_df:
            X = X.iloc[:, indices]
        else:
            X = X[:, indices]
        return X

    def _get_tree_num(self, n_feat):
        depth = self.estimator.get_params()['max_depth']
        if depth == None:
            depth = 10
        # how many times a feature should be considered on average
        f_repr = 100
        # n_feat * 2 because the training matrix is extended with n shadow features
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        n_estimators = int(multi * f_repr)
        return n_estimators

    def _get_imp(self, X, y):
        try:
            self.estimator.fit(X, y)
        except Exception as e:
            raise ValueError('Please check your X and y variable. The provided '
                             'estimator cannot be fitted to your data.\n' + str(e))
        try:
            imp = self.estimator.feature_importances_
        except Exception:
            raise ValueError('Only methods with feature_importance_ attribute '
                             'are currently supported in BorutaPy.')
        return imp

    def _get_shuffle(self, seq):
        self.random_state.shuffle(seq)
        return seq

    def _add_shadows_get_imps(self, X, y, dec_reg):
        # find features that are tentative still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = np.copy(X[:, x_cur_ind])
        x_cur_w = x_cur.shape[1]
        # deep copy the matrix for the shadow matrix
        x_sha = np.copy(x_cur)
        # make sure there's at least 5 columns in the shadow matrix for
        while (x_sha.shape[1] < 5):
            x_sha = np.hstack((x_sha, x_sha))
        # shuffle xSha
        x_sha = np.apply_along_axis(self._get_shuffle, 0, x_sha)
        # get importance of the merged matrix
        imp = self._get_imp(np.hstack((x_cur, x_sha)), y)
        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]
        imp_real = np.zeros(X.shape[1])
        imp_real[:] = np.nan
        imp_real[x_cur_ind] = imp[:x_cur_w]
        return imp_real, imp_sha

    def _assign_hits(self, hit_reg, cur_imp, imp_sha_max):
        # register hits for features that did better than the best of shadows
        cur_imp_no_nan = cur_imp[0]
        cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
        hits = np.where(cur_imp_no_nan > imp_sha_max)[0]
        hit_reg[hits] += 1
        return hit_reg

    def _do_tests(self, dec_reg, hit_reg, _iter):
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()

        if self.two_step:
            # two step multicor process
            # first we correct for testing several features in each round using FDR
            to_accept = self._fdrcorrection(to_accept_ps, alpha=self.alpha)[0]
            to_reject = self._fdrcorrection(to_reject_ps, alpha=self.alpha)[0]

            # second we correct for testing the same feature over and over again
            # using bonferroni
            to_accept2 = to_accept_ps <= self.alpha / float(_iter)
            to_reject2 = to_reject_ps <= self.alpha / float(_iter)

            # combine the two multi corrections, and get indexes
            to_accept *= to_accept2
            to_reject *= to_reject2
        else:
            # as in th original Boruta, we simply do bonferroni correction
            # with the total n_feat in each iteration
            to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
            to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))

        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        return dec_reg

    def _fdrcorrection(self, pvals, alpha=0.05):
        """
        Benjamini/Hochberg p-value correction for false discovery rate, from
        statsmodels package. Included here for decoupling dependency on statsmodels.
        Parameters
        ----------
        pvals : array_like
            set of p-values of the individual tests.
        alpha : float
            error rate
        Returns
        -------
        rejected : array, bool
            True if a hypothesis is rejected, False if not
        pvalue-corrected : array
            pvalues adjusted for multiple hypothesis testing to limit FDR
        """
        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
        nobs = len(pvals_sorted)
        ecdffactor = np.arange(1, nobs + 1) / float(nobs)

        reject = pvals_sorted <= ecdffactor * alpha
        if reject.any():
            rejectmax = max(np.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        # reorder p-values and rejection mask to original order of pvals
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_

    def _nanrankdata(self, X, axis=1):
        """
        Replaces bottleneck's nanrankdata with scipy and numpy alternative.
        """
        ranks = sp.stats.mstats.rankdata(X, axis=axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _check_params(self, X, y):
        """
        Check hyperparameters as well as X and y before proceeding with fit.
        """
        # check X and y are consistent len, X is Array and y is column
        X, y = check_X_y(X, y)
        if self.perc <= 0 or self.perc > 100:
            raise ValueError('The percentile should be between 0 and 100.')

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('Alpha should be between 0 and 1.')

    def _print_results(self, dec_reg, _iter, flag):
        n_iter = str(_iter) + ' / ' + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ['Iteration: ', 'Confirmed: ', 'Tentative: ', 'Rejected: ']

        # still in feature selection
        if flag == 0:
            n_tentative = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            if self.verbose == 1:
                output = cols[0] + n_iter
            elif self.verbose > 1:
                output = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])

        # Boruta finished running and tentatives have been filtered
        else:
            n_tentative = np.sum(self.support_weak_)
            n_rejected = np.sum(~(self.support_ | self.support_weak_))
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
            output = "\n\nBorutaPy finished running.\n\n" + result
        print(output)


from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator
    .. versionadded:: 0.18
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    max_train_size : int, default=None
        Maximum size for a single training set.
    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed value
        with ``gap=0``.
    gap : int, default=0
        Number of samples to exclude from the end of each train set before
        the test set.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TimeSeriesSplit()
    >>> print(tscv)
    TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    >>> for train_index, test_index in tscv.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    TRAIN: [0 1 2 3] TEST: [4]
    TRAIN: [0 1 2 3 4] TEST: [5]
    >>> # Fix test_size to 2 with 12 samples
    >>> X = np.random.randn(12, 2)
    >>> y = np.random.randint(0, 2, 12)
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3 4 5] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
    TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10 11]
    >>> # Add in a 2 period gap
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2, gap=2)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5] TEST: [8 9]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [10 11]
    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i``th split,
    with a test set of size ``n_samples//(n_splits + 1)`` by default,
    where ``n_samples`` is the number of samples.
    """

    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None,
                 test_size=None,
                 gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = self.test_size if self.test_size is not None \
            else n_samples // n_folds

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                (f"Cannot have number of folds={n_folds} greater"
                 f" than the number of samples={n_samples}."))
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                (f"Too many splits={n_splits} for number of samples"
                 f"={n_samples} with test_size={test_size} and gap={gap}."))

        indices = np.arange(n_samples)
        test_starts = range(n_samples - n_splits * test_size,
                            n_samples, test_size)

        for test_start in test_starts:
            train_end = test_start - gap
            if self.max_train_size and self.max_train_size < train_end:
                yield (indices[train_end - self.max_train_size:train_end],
                       indices[test_start:test_start + test_size])
            else:
                yield (indices[:train_end],
                       indices[test_start:test_start + test_size])


def showsplit(df, cv, y=None):
    figsflask = not isnotebook()
    res = {}
    import numbers
    if isinstance(cv, numbers.Integral):
        return DF({"cv": cv}, index=[0])

    if y is None:
        df = DF(index=df.index)
    else:
        df = df[y]
    i = 0
    resd = []
    for train_index, test_index in cv.split(df.values):
        resd.append({'train_start': train_index.min() / len(df), 'train_end': train_index.max() / len(df),
                     'test_start': test_index.min() / len(df), 'test_end': test_index.max() / len(df)})
        df['test' + str(i)] = np.nan
        df['train' + str(i)] = np.nan
        df['train' + str(i)].iloc[train_index] = i * 0.1
        df['test' + str(i)].iloc[test_index] = i * 0.1
        i += 1
    if figsflask: img = plotstart()

    df.plot(marker='.')
    if figsflask:
        res['figs'] = [plotend(img)]
    res['df'] = DF(resd).myround2()
    return res


def fs1(res, y, xs=None):
    if xs is None:
        xs = res.columns
    # print(y)
    xs = list(xs)
    xs = list(set(xs) - set([y]))
    df = res[xs + [y]]
    # df=res[xs].fillna(res[xs].mean()) #+[y] if duplicate
    # ftest=DF({'f':f_classif(df[xs],df[y])[0]},index=xs)
    # chi2abs,chi2pvals=chi2(df[xs].abs(),df[y])
    dcors = {}
    pearsons = {}
    fctests = {}
    frtests = {}
    chi2abs = {}
    mir = {}
    mic = {}
    kendalls = {}
    for col in xs:
        dfdropna = df[[col, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if dfdropna.shape == (0, 2):
            continue
        try:
            fctests[col] = f_classif(dfdropna[[col]], dfdropna[y])[0]
        except:
            pass

        # frtests[col]=f_regression(dfdropna[[col]],dfdropna[y])[0]
        try:
            pass
            # mic[col]=mutual_info_classif(dfdropna[[col]],dfdropna[y])
        except:
            pass
        # mir[col]=mutual_info_regression(dfdropna[[col]],dfdropna[y])
        try:  # only discete
            pass
            # chi2abs[col]=chi2(dfdropna[[col]],dfdropna[y])[0]
        except:
            pass
        # dfdropna
        dcors[col] = dcor(dfdropna[col], dfdropna[y])
        pearsons[col] = dfdropna[[col, y]].corr().iloc[1, 0]
        # kendalls[col]=dfdropna[[col,y]].corr(method='kendall').iloc[1,0]

    corrs = DF(pearsons, index=['pearsonabs']).abs().T.join(DF(dcors, index=['dcor']).T).join(
        DF(fctests, index=['fc']).T)
    # .join(DF(mic,index=['mic']).T).join(DF(mir,index=['mir']).T).\
    # join(DF(frtests,index=['fr']).T).join(DF(chi2abs,index=['chi2c']).T).join(DF(kendalls,index=['kendall']).T)
    res = pd.concat([corrs, corrs.rank(pct=True).add_prefix('rk.')], axis=1).sort_values(by='rk.dcor', ascending=False)
    logging.info(f"fs1 y={y} rk.dcor index {list(res['rk.dcor'].index)}")
    #    logging.info(f"fs1 rk.pearsonabs [{','.join(res['rk.pearsonabs'])}]")
    return res


def plotroc(ytrue, ypredproba):
    p, r, threshold = precision_recall_curve(ytrue, ypredproba)
    fpr, tpr, threshold = roc_curve(ytrue, ypredproba)
    plt.figure()
    # print(f"thresh={threshold}")
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC= %0.2f' % auc(fpr, tpr))
    baselinepr = (ytrue == 1).sum() / (ytrue == 0).sum()
    plt.plot(p, r, color='red', lw=2,
             label=f'AP={round(average_precision_score(ytrue, ypredproba), 2)}) baseline={round(baselinepr, 2)}')

    print(f"baselinepr={baselinepr}")

    plt.hlines(baselinepr, 0, 1, color='navy', lw=2, linestyle='--')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate,recall')
    plt.ylabel('True Positive Rate,precision')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plotshaps(model, df, xs, y, request=None):
    figsflask = not isnotebook()
    res = {}
    figs = []
    bigfigs = []
    if 'Pipe' in model.__class__.__name__:
        if 'xgb' in model.named_steps['pipe1'].__class__.__name__.lower() and model.named_steps['pipe1'].get_params()[
            'booster'] in ('gbtree', None):
            explainer = shap.TreeExplainer(model.named_steps['pipe1'], model_output='raw')
        elif 'NN' in model.named_steps['pipe1'].__class__.__name__:
            explainer = shap.DeepExplainer(model.named_steps['pipe1'].model.model,
                                           data=df[xs].values)  # , session=None, learning_phase_flags=None)
        elif 'Logistic' in model.named_steps['pipe1'].__class__.__name__:
            explainer = shap.KernelExplainer(model.named_steps['pipe1'].predict_proba, data=df[xs].values, link='logit',
                                             l1_reg='aic')
        else:
            raise ValueError(
                f"shap for model {model.named_steps['pipe1'].__class__.__name__} {model.named_steps['pipe1'].get_params()} not imlemented in fs.py")
    else:
        if 'randomforest' in model.__class__.__name__.lower():
            explainer = shap.TreeExplainer(model, model_output='raw')
        elif 'xgb' in model.__class__.__name__.lower() and model.get_params()['booster'] in ('gbtree', None):

            # booster = model.get_booster()
            # model_bytearray = booster.save_raw()[4:]
            # print(model_bytearray[:10])
            # booster.save_raw = lambda : model_bytearray

            mybooster = model.get_booster()
            model_bytearray = mybooster.save_raw()[4:]

            def myfun(self=None):
                return model_bytearray

            mybooster.save_raw = myfun

            #            model = xgb.XGBClassifier()
            #             booster = model.get_booster()
            #             model2 = booster.save_raw()[4:]
            #             booster.save_raw = lambda: model2

            explainer = shap.TreeExplainer(mybooster)

            # values = explainer.shap_values(test_x, test_y)

            # explainer=shap.TreeExplainer(booster)
            # explainer=shap.TreeExplainer(model,model_output='raw')
        elif 'NN' in model.__class__.__name__:
            explainer = shap.DeepExplainer(model.model.model,
                                           data=df[xs].values)  # , session=None, learning_phase_flags=None)
        elif 'Logistic' in model.__class__.__name__:
            explainer = shap.KernelExplainer(model.predict_proba, data=df[xs].values, link='logit', l1_reg='aic')
        else:
            raise ValueError(
                f"shap for model {model.__class__.__name__} and params={model.get_params()} not imlemented in fs.py")

    #             indices = np.argsort(global_importances)[::-1]
    #             features_ranked = []
    #             for f in range(df[xs].shape[1]):
    #                 features_ranked.append(xs[indices[f]])
    shap_values = explainer.shap_values(df[xs].values)  # , tree_limit=5)
    concat = np.concatenate(shap_values) if type(shap_values) == type([]) else shap_values
    global_importances = np.nanmean(np.abs(concat), axis=0)
    dfshaps = DF({'shap': global_importances}, index=xs).sort_values(by='shap', ascending=False)
    res['xsshaps'] = list(dfshaps.index)
    shapdf = DF({'shap': global_importances}, index=xs).rank(ascending=True, pct=True)
    res['shapdf'] = shapdf
    if figsflask: img = plotstart()
    shap.summary_plot(shap_values, df[xs], plot_type="bar", class_names=model.classes_)
    plt.show()

    if figsflask: figs.append(plotend(img))
    #     try:
    #         plot_importance(model,importance_type='gain')
    #         if figsflask: figs.append(plotend(img))
    #         display(DF({'fi':model.feature_importances_},index=xs).join(dfshaps))
    #         print("fs1 df")
    #         display(fs1(df,y,xs))
    #     except Exception as e:
    #         print(e)

    res['html'] = ''
    if 'shapforce' in request:

        shap.initjs()
        try:
            for i in range(len(explainer.expected_value)):
                if figsflask:
                    forceplot = shap.force_plot(explainer.expected_value[i], shap_values[i], df[xs])
                    s = StringIO()
                    shap.save_html(s, forceplot)
                    res['html'] = s.getvalue()
                else:
                    display(shap.force_plot(explainer.expected_value[i], shap_values[i], df[xs]))
        except Exception as e:
            print(f"forceplot exception:{e}")

    res['figs'] = figs

    return res


def runscoring(df, y, xs, model, nansy, nansx, scoring=None, request=None, verbose=0, cv=None, dftest=None):
    # df is dftrain
    figsflask = not isnotebook()
    eval_metric = scoring

    try:
        if 'xgb' in model.named_steps['pipe1'].__class__.__name__.lower():
            if eval_metric == 'f1':
                eval_metric = minusf1
    except:
        if 'xgb' in 'xgb' in model.__class__.__name__.lower():
            if eval_metric == 'f1':
                eval_metric = minusf1

    ytype = type_of_target(df[y])

    res = {}
    figs = []
    bigfigs = []
    xs = list(xs)
    if type(model) == type("string"):
        modelstr = model
        model = eval(model)
    else:
        modelstr = str(model)
    res['seed'] = model.get_params().get('random_state', None)
    #     try:
    #         res['freq']=pd.infer_freq(df[df.coin_id==df.iloc[0].coin_id].index)
    #         res['coin_ids']=df.coin_id.unique().astype(int)
    #     except Exception as e:
    #         print(f"{str(e)}")

    res['timemin'] = df.index.min()
    res['timemax'] = df.index.max()
    ytrain = eval('df[y]' + nansy)
    xtrain = eval('df[xs]' + nansx) if nansx is not None else df[xs]

    if dftest is not None:
        # res['coin_ids_test']=dftest.coin_id.unique().astype(int)
        res['timemin_test'] = dftest.index.min()
        res['timemax_test'] = dftest.index.max()
        ytest = eval('dftest[y]' + nansy)
        xtest = eval('dftest[xs]' + nansx) if nansx is not None else dftest[xs]

    print(f"typemodelname={type(model).__name__}")

    if not (('XGB' in type(model).__name__) or ('NN' in type(
            model).__name__)):  # not in ['XGBRegressor' , 'XGBClassifier']: #XGBRegressor  run with and without nans
        model.fit(xtrain, ytrain)
    else:
        eval_set = [(xtrain, ytrain)] if dftest is None else [(xtrain, ytrain), (xtest, ytest)]
        model.fit(xtrain, ytrain, eval_metric=eval_metric, eval_set=eval_set, verbose=verbose)
        if 'evals_result' in request:
            # ipdb.set_trace()
            # print("evals_result on df, dftest")
            if figsflask: img = plotstart()
            DF(model.evals_result()[
                   'validation_0']).plot()  # check that model error reduces significantly i.e. not from 0.28 to 0.27 (i.e. model is almost worthless)
            plt.show()
            if figsflask: figs.append(plotend(img))
            try:
                if figsflask: img = plotstart()
                DF(model.evals_result()[
                       'validation_1']).plot()  # check that model error reduces significantly i.e. not from 0.28 to 0.27 (i.e. model is almost worthless)
                plt.show()
                if figsflask: figs.append(plotend(img))
            except:
                pass

    if hasattr(model, 'feature_importances_'):
        coef = model.feature_importances_
    elif hasattr(model, 'coef_'):
        coef = model.coef_
    else:
        coef = np.ones(len(xs))
    fssorted = list(DF(np.abs(coef.flatten()), index=xs).sort_values(by=0, ascending=False).index)
    ypredis = model.predict(xtrain).flatten()
    # odel.classes_
    ypredprobais = model.predict_proba(xtrain)[:,
                   1]  # . The binary case expects a shape (n_samples,), and the scores must be the scores of the class with the greater label
    res['ymeanis'] = np.nanmean(ytrain)
    res['scoreis'] = get_scorer(scoring)._score_func(ytrain, ypredis)
    if is_regressor(model):
        res['mseis'] = mean_squared_error(ytrain, ypredis)
        res['r2is'] = r2_score(ytrain, ypredis)
    else:

        res['baccis'] = balanced_accuracy_score(ytrain, ypredis)
        res['kappais'] = cohen_kappa_score(ytrain, ypredis)
        try:
            res['apis'] = average_precision_score(ytrain, ypredprobais, average='weighted')
            res['pis'] = precision_score(ytrain, ypredis)
            res['recallis'] = recall_score(ytrain, ypredis)
        except:
            pass

        res['f1is'] = f1_score(ytrain, ypredis, average='binary')  # only for 1
        # res['f1is']=f1_score(ytrain,ypredis,average='weighted')

    if 'xshist' in request:
        if figsflask: img = plotstart()
        sns.pairplot(xtrain.join(pd.Series(ypredis, name='ypredis', index=df.index)), hue='ypredis', diag_kind='hist')
        # sns.pairplot(df[xs].join(,hue=model.predict(xtrain))
        #         for x in xs:
        #             sns.histplot(df[x],hue=model.predict(xtrain))
        if figsflask: bigfigs.append(plotend(img))

    if dftest is not None:

        # ipdb.set_trace()
        ypred = model.predict(xtest).flatten()

        if 'xshist' in request:
            if figsflask: img = plotstart()
            sns.pairplot(xtest.join(pd.Series(ypred, name='ypred', index=dftest.index)), hue='ypred', diag_kind='hist')
            if figsflask: bigfigs.append(plotend(img))

        ypredproba = model.predict_proba(xtest)[:, 1]  # probas of 1  classes are maybe sorted
        # ypredproba1=model.predict_proba(xtest)[:,0]
        res['ymean'] = np.nanmean(ytest)
        res['score'] = get_scorer(scoring)._score_func(ytest, ypred)
        if is_regressor(model):
            res['mse'] = mean_squared_error(ytest, ypred)
            res['r2'] = r2_score(ytest, ypred)
        else:

            res['bacc'] = balanced_accuracy_score(ytest, ypred)
            res['kappa'] = cohen_kappa_score(ytest, ypred)
            # res['f1']=f1_score(ytest,ypred,average='weighted')
            res['f1'] = f1_score(ytest, ypred, average='binary')

            if ytype == 'binary':
                res['auc'] = roc_auc_score(ytest, ypredproba)
                res['ap'] = average_precision_score(ytest, ypredproba, average='weighted')
                res['p'] = precision_score(ytest, ypred)
                res['recall'] = recall_score(ytest, ypred)

        if 'roc' in request and ytype == 'binary':

            if figsflask: img = plotstart()
            plotroc(ytest, ypredproba)
            if figsflask: figs.append(plotend(img))
            # plotroc(ytest,ypredproba1)

        for k in range(1, 10):
            sk = str(k)
            try:
                pnlb = np.nanmean(dftest['pnlb+' + sk][ypred == 1])
                pnlbstd = np.nanstd(dftest['pnlb+' + sk][ypred == 1])
                res['pnlb+' + sk] = pnlb
                res['pnlbstd' + sk] = pnlbstd
                if 'pnlhist' in request:
                    if figsflask: img = plotstart()
                    dftest['pnlb+' + sk].hist()
                    if figsflask: figs.append(plotend(img))
            except:
                pass

        if 'nnplots' in request:
            if figsflask: img = plotstart()
            nnplotloss(model.history)
            if figsflask: figs.append(plotend(img))
            if figsflask: img = plotstart()
            nnplotmetrics(model.history)
            if figsflask: figs.append(plotend(img))

    if 'cm' in request:  # confusion matrix
        for thresh in [0, 0.50, 0.7, 0.9]:  # chnage to 5% , 10% quantiles of proba or proba distribution

            # OS

            ypredproba = model.predict_proba(dftest[xs])
            ypred1 = model.predict(dftest[xs])
            ypredmax = model.classes_.take(np.argmax(ypredproba, axis=1), axis=0)
            ypred = np.where(np.max(ypredproba, axis=1) > thresh, ypredmax, 0)
            ytrue = dftest[y]
            print(f"thres={thresh}, modelclass{model.classes_}  {type(model.classes_)}")
            cmat = DF(confusion_matrix(ytrue, ypred, labels=model.classes_),
                      columns=model.classes_.astype('str') + npa(['p'], dtype=np.object),
                      index=model.classes_.astype('str') + npa(['t'], dtype=np.object))
            res[f'cm{thresh}'] = cmat
            # cmat1=cmat.iloc[:2,:2]
            if figsflask: display(cmat)
            # print(f"bacc={balanced_accuracy_score(ytrue,ypred)}")
            # f1=f1_score(ytrue,ypred,average='macro',labels=[-1,1])
            #             print(f"my prec1={cmat.loc['1t','1p']/cmat['1p'].sum()}")
            #             print(f"my recall1={cmat.loc['1t','1p']/cmat.loc['1t'].sum()}")

            #             print(f"my prec1={cmat1.loc['1t','1p']/cmat1['1p'].sum()}")
            #             print(f"my recall1={cmat1.loc['1t','1p']/cmat1.loc['1t'].sum()}")

            #             print(f"  f1 macro labels11={f1}"   )

            res[f'cr{thresh}'] = classification_report(ytrue, ypred, labels=model.classes_, output_dict=False)  # [-1,1]
            if isnotebook():
                display(res[f'cm{thresh}'])
                print(res[f'cr{thresh}'])

            ## IS

            ypredproba = model.predict_proba(xtrain)  # dftrain[xs]
            ypred1 = model.predict(xtrain)
            ypredmax = model.classes_.take(np.argmax(ypredproba, axis=1), axis=0)
            ypred = np.where(np.max(ypredproba, axis=1) > thresh, ypredmax, 0)
            ytrue = ytrain  # dftrain[y]
            print(f"is thres={thresh}, modelclass{model.classes_}  {type(model.classes_)}")
            cmat = DF(confusion_matrix(ytrue, ypred, labels=model.classes_),
                      columns=model.classes_.astype('str') + npa(['p'], dtype=np.object),
                      index=model.classes_.astype('str') + npa(['t'], dtype=np.object))
            res[f'iscm{thresh}'] = cmat
            # cmat1=cmat.iloc[:2,:2]
            if figsflask: display(cmat)
            res[f'iscr{thresh}'] = classification_report(ytrue, ypred, labels=model.classes_,
                                                         output_dict=False)  # [-1,1]
            if isnotebook():
                display(res[f'iscm{thresh}'])
                print(res[f'iscr{thresh}'])

    if 'pdp' in request:
        for feature in xs:
            # https://github.com/SauceCat/PDPbox/blob/master/tutorials/pdpbox_binary_classification.ipynb
            if figsflask: img = plotstart()
            fig, axes, summary_df = info_plots.target_plot(df=df[xs + [y]], feature=feature, feature_name=feature,
                                                           target=y, show_percentile=True)
            _ = axes['bar_ax']  # .set_xticklabels(['Female', 'Male'])
            if figsflask: figs.append(plotend(img))
            if figsflask: img = plotstart()
            fig, axes, summary_df = info_plots.actual_plot(model=model, X=df[xs], feature=feature, feature_name=feature,
                                                           predict_kwds={})
            if figsflask: figs.append(plotend(img))
            pdp_f = pdp.pdp_isolate(model=model, dataset=df, model_features=xs, feature=feature, predict_kwds={})
            if figsflask: img = plotstart()
            fig, axes = pdp.pdp_plot(pdp_f, feature,
                                     plot_pts_dist=True)  # frac_to_plot=0.5, plot_lines=True, x_quantile=True, show_percentile=True, plot_pts_dist=True
            _ = axes['pdp_ax']  # .set_xticklabels(['Female', 'Male'])
            if figsflask: figs.append(plotend(img))
    if 'shaps' in request:
        print("importance on df")
        try:
            plotshaps(model, df, xs, y, request)
        except Exception as e:
            print(str(e))

        if dftest is not None:
            print("importance on fitted dftest")
            modeltest = clone(model)
            modeltest.fit(xtest, ytest)
            plotshaps(modeltest, dftest, xs, y, request)
            plt.show()

    if cv is not None:
        res['cv'] = [cross_val_score(model, df[xs], y=df[y], scoring=scoring, cv=cv)]

    # output xs sorted by feature importance , coinids sorted, model strings   time1  time2
    res['figs'] = figs
    return {**res, **{'y': y, 'xs': fssorted, 'nansy': nansy, 'nansx': nansx,
                      'model': modelstr.replace('linear_model.', '').replace('xgb.', '')}}


def runselector(df, y, xs, model, nansy, nansx, request=None, scoring=None, verbose=0, cv=None, eliniter=5,
                dftest=None):
    '''
    >>> df=DF({'x1':[1,2,3,4,5],'x2':[0,0,0,0,0]})
    
    '''
    ytype = type_of_target(df[y])
    figsflask = not isnotebook()
    dfs = []
    res = DF()
    htmls = []
    figs = []
    ifigs = []
    bigfigs = []

    eval_metric = scoring

    try:
        if 'xgb' in model.named_steps['pipe1'].__class__.__name__.lower():
            if eval_metric == 'f1':
                eval_metric = minusf1
    except:
        if 'xgb' in 'xgb' in model.__class__.__name__.lower():
            if eval_metric == 'f1':
                eval_metric = minusf1

    if request is None:
        request = ['boruta', 'rfe', 'sfsb', 'sfsf', 'shap', 'eli']

    xs = list(xs)
    if scoring is None:
        if df[y].nunique() < 10:
            scoring = 'balanced_accuracy'
        else:
            scoring = 'r2'
        print('runselector scoring is None.using {}'.format(scoring))

    ytrain = eval('df[y]' + nansy)
    xtrain = eval('df[xs]' + nansx) if nansx is not None else df[xs]
    try:
        inferfreq = pd.infer_freq(df.index)
    except:
        inferfreq = None

    logging.info(
        f"runselector START lendf = {len(df)} df.index.min,max={df.index.min(), df.index.max()} dftest.minmax={dftest.index.min(), dftest.index.max() if dftest is not None else 'None'}  inferfreq={inferfreq} meansecsdiff= {float(np.diff(npa(df.index)).mean()) / 1e9}secs scoring={scoring} \n modelclass={model.__class__.__name__} modeldict={model.__dict__} xs={xs} \n {df.describe()}")

    try:
        model = eval(model)
    except Exception as e:
        print(e)

    fs1df = fs1(df, y, xs)
    res = fs1df[fs1df.columns[fs1df.columns.str.contains('rk\\.')]]

    if 'boruta' in request:
        boruta_selector = BorutaPy(model).fit(xtrain, ytrain)  # , n_estimators = 10, random_state = 0)
        #      boruta_selector=BorutaPy(model).fit(xtrain.values,ytrain.values)#, n_estimators = 10, random_state = 0)
        boruta = DF({'boruta': boruta_selector.ranking_, 'xs': xs}).set_index('xs').rank(ascending=False,
                                                                                         pct=True).sort_values(
            by='boruta', ascending=False)
        logging.info(f'boruta: {boruta.round(2) * 100}')
        res = res.join(boruta)
    # boruta_selector = selectormodel

    if 'abscoef' in request:
        try:
            modelcoef = clone(model)
            modelcoef.fit(xtrain, ytrain)
            rfeselectorranking = np.abs(modelcoef.coef_[
                                            0]) * xtrain.std()  # multiply by feature stddev, ocnditional that feture is centered at 0
            abscoef = DF({'abscoef': rfeselectorranking, 'xs': xs}).set_index('xs').rank(ascending=False, pct=True)
            res = res.join(abscoef)
            abscoeflog = DF({'abscoefbystd': rfeselectorranking, 'coef': modelcoef.coef_[0], 'std': xtrain.std(),
                             'xs': xs}).set_index('xs').sort_values(by='abscoefbystd', ascending=False)
            logging.info(f'abscoef:\n {abscoeflog}')
        except Exception as e:
            print(f"!abscoef:{e}")

    if 'rfe' in request:
        try:
            rfeselector = RFE(model, 1, step=1).fit(xtrain, ytrain)
            rfeselectorranking = rfeselector.ranking_
            rfe = DF({'rfe': rfeselectorranking, 'xs': xs}).set_index('xs').rank(ascending=False, pct=True)
            res = res.join(rfe)
        except Exception as e:
            print(f"rfe:{e}")
    if 'sfsf' in request:
        sfsf = SequentialFeatureSelector(model, k_features=len(xs), forward=True, floating=False, verbose=0,
                                         scoring=scoring, cv=cv).fit(xtrain, ytrain, custom_feature_names=xs)
        sfsF = DF(np.unique(DF(sfsf.get_metric_dict()).T['feature_names'].sum(), return_counts=True)).T.set_index(
            0).rank(pct=True).sort_values(by=1, ascending=False).rename(columns={1: 'sfsF'})
        if verbose > 1:
            display(DF(sfsf.get_metric_dict()).T[['avg_score', 'cv_scores', 'std_dev', 'feature_names']])
        res = res.join(sfsF)
        logging.info(f'sfsf:\n {sfsF.round(2) * 100}')
    if 'sfsb' in request:
        sfsb = SequentialFeatureSelector(model, k_features=1, forward=False, floating=False, verbose=0, scoring=scoring,
                                         cv=cv).fit(xtrain, ytrain, custom_feature_names=xs)
        sfsB = DF(np.unique(DF(sfsb.get_metric_dict()).T['feature_names'].sum(), return_counts=True)).T.set_index(
            0).rank(pct=True).sort_values(by=1, ascending=False).rename(columns={1: 'sfsB'})

        if verbose > 1:
            sfsbd = DF(sfsb.get_metric_dict()).T[['avg_score', 'cv_scores', 'std_dev', 'feature_names']]
            if dftest is not None:
                sfsbd['dftest'] = ''

                for i, row in sfsbd.iterrows():
                    #                    modelfit=model.fit(df[list(row['feature_names'])],df[y],eval_metric=eval_metric,eval_set=[(dftest[list(row['feature_names'])], dftest[y])],verbose=0)
                    #                    ipdb.set_trace()
                    if 'Pipe' in model.__class__.__name__:
                        model.fit(df[list(row['feature_names'])], df[y], pipe1__eval_metric=eval_metric,
                                  pipe1__eval_set=[(dftest[list(row['feature_names'])], dftest[y])], pipe1__verbose=0)
                        sfsbd.at[i, 'dftest'] = model.named_steps['pipe1'].evals_result()['validation_0']
                    else:
                        model.fit(df[list(row['feature_names'])], df[y], eval_metric=eval_metric,
                                  eval_set=[(dftest[list(row['feature_names'])], dftest[y])], verbose=0)
                        sfsbd.at[i, 'dftest'] = model.evals_result()['validation_0']

                    print(list(row['feature_names']), eval_metric, sfsbd.at[i, 'dftest'])
            # {'validation_0': {'logloss': ['0.604835', '0.531479']},
            logging.info(f"sfsbd=\n{sfsbd.myround2()}")
            #            sfsbd.to_pickle('dfsfs.pkl')
            display(sfsbd.round(3))
        logging.info(f'sfsb:\n {sfsB.round(2) * 100}')
        res = res.join(sfsB)
    resd = {}
    rmres = {}
    if 'runmodel' in request:
        rmres = runscoring(df=df, y=y, xs=xs, model=model, nansy=nansy, nansx=nansx, scoring=scoring,
                           eval_metric=eval_metric, verbose=10, cv=cv, dftest=dftest, request=request)
        rmres = {f'rm.{k}': v for k, v in rmres.items()}
        # resd={**res,**rmres1}
    #    model.fit(xtrain.values,ytrain.values)
    model.fit(xtrain, ytrain)

    if 'eli' in request:
        if cv is None:
            cv = 'prefit'
        permuter = PermutationImportance(model, scoring=None, cv=cv, n_iter=eliniter,
                                         random_state=42)  # instantiate permuter object #'balanced_accuracy'  'prefit'
        elidf = DF({'eli': permuter.fit(xtrain.values, ytrain.values).feature_importances_, 'xs': xs}).set_index(
            'xs').rank(ascending=True, pct=True).sort_values(by='eli', ascending=False)
        logging.info(f'eli: {elidf.round(2) * 100}')
        res = res.join(elidf)

    if 'shap' in request:

        try:
            resshaps = plotshaps(model, df, xs, y)
            res = res.join(resshaps['shapdf'])
            figs += resshaps['figs']
            htmls.append(resshaps['html'])
        except Exception as e:
            print(str(e))

    res['mean'] = res.mean(axis=1)
    res = res.sort_values(by='mean', ascending=False)

    if 'corr' in request:
        # logging.info(f"res.corr.mean=\n{100*res.corr().round(2)}")

        # print(f"meancorr=\n{res.corr().mean().round(2)*100}")
        resd['corr.mean'] = 100 * res.corr().round(2)

        # logging.info(f"meancorr=\n{res.corr().mean().myround2()}")
        res1 = res.copy()
        res1['pfname'] = res1.index.str.split('.').str[-1]
        res1 = res1.groupby('pfname').mean().sort_values(by='mean', ascending=False)  # .set_index('pfname')
        resd['corr.pf'] = res1.myround2()
        # logging.info(f"pure fs mean rank\n {res1.myround2()}")
        # print("pure features wtau")

        resd['corr.pearson'] = 100 * res.corr().round(2)
        if isnotebook():
            display(resd['corr.pearson'])
            display(res1)

        try:
            if isnotebook():
                print("wtau")
                display(100 * res1.calccorr(method='wtau').round(2))

            resd['corr.wtau'] = res1.calccorr(method='wtau').myround2()
            logging.info(f"pure features wtau \n{res1.calccorr(method='wtau').myround2()}")
        except Exception as e:
            print(f"wtau calcorr exception{str(e)}")

    logging.info(f"runselector complete res=\n{res.round(2) * 100} {list(res.index)}")

    # resd={}

    resd['res'] = res
    resd['htmls'] = htmls
    resd['figs'] = figs
    resd['bigfigs'] = bigfigs

    return resd


def nnplotloss2(history):
    label = ''
    n = 0
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)

    if 'val_loss' in history.history:
        plt.semilogy(history.epoch, history.history['val_loss'],
                     color=colors[n], label='Val ' + label,
                     linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss -train --test')
    plt.legend()


def nnplotmetrics2(history):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        if 'val_loss' in history.history:
            plt.plot(history.epoch, history.history['val_' + metric], color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()


def nnplotroc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def nnplotroc2(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def nnplotloss(history):
    plt.semilogy(history.epoch, history.history['loss'], label='Train ')
    if 'val_loss' in history.history:
        plt.semilogy(history.epoch, history.history['val_loss'], label='Val ', linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def nnplotmetrics(history):
    print(f"avail metrics:{history.history.keys()}")
    metrics = [x.replace('val_', '') for x in history.history.keys() if 'val_' in x]
    for n, metric in enumerate(metrics):
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label='train')
        if 'val_loss' in history.history:
            plt.plot(history.epoch, history.history['val_' + metric], linestyle="--", label='val')
        plt.xlabel('epoch')
        plt.ylabel(metric)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])

    plt.legend()
    plt.show()


try:
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_recall', verbose=1, patience=20, mode='max',
                                                      restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=40, min_lr=0.000001,
                                                     verbose=0)
except:
    print("err init tf METRICS")


def build_model(nhidden, nfirst, drop1, drop2):
    output_bias = tf.keras.initializers.Constant(np.log([0.1]))
    model = keras.Sequential([keras.layers.Dense(nfirst, activation='relu', input_shape=(len(xs),)),
                              keras.layers.Dropout(drop1),
                              keras.layers.Dense(nhidden, activation='relu'),
                              keras.layers.Dropout(drop2),
                              keras.layers.Dense(nhidden, activation='relu'),
                              keras.layers.Dropout(drop2),
                              keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
                              ])
    model.compile(optimizer=keras.optimizers.Adam(lr=0.005), loss=keras.losses.BinaryCrossentropy(), metrics=METRICS)

    return model


from sklearn.base import BaseEstimator, ClassifierMixin


class MyLogisticRegression(LogisticRegression):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C,
                         fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                         random_state=random_state, solver=solver, max_iter=max_iter,
                         multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                         l1_ratio=l1_ratio)

    def fit(self, X, y, sample_weight=None, eval_metric=None, eval_set=None, verbose=0):
        super().fit(X, y, sample_weight)
        if eval_metric is not None:
            scorer = SCORERS[eval_metric]._score_func
            self.score = scorer(eval_set[0][1], self.predict(eval_set[0][0]))

    def evals_result(self):
        return {'validation_0': self.score}


#    def __call__(self,X):
#        return super().predict_proba(X) #for shapley kernel explainer   need proba maybe?


class MyLasso(Lasso):  # regression
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic', classfunc=np.sign):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize,
                         precompute=precompute, copy_X=copy_X, max_iter=max_iter,
                         tol=tol, warm_start=warm_start, positive=positive,
                         random_state=random_state, selection=selection)

    def fit(self, X, y, sample_weight=None, eval_metric=None, eval_set=None, verbose=0):
        super().fit(X, y, sample_weight)
        if eval_metric is not None:
            scorer = SCORERS[eval_metric]._score_func
            self.score = scorer(classfunc(eval_set[0][1]), classfunc(self.predict(eval_set[0][0])))

    def evals_result(self):
        return {'validation_0': self.score}


def build_model2(nfirst, nfeatures, nhidden1, nhidden2, dropout, output_bias, lr):  # 0.1
    output_bias = tf.keras.initializers.Constant(np.log([output_bias]))
    model = keras.Sequential(
        [keras.layers.Dense(nfirst, activation='relu', input_shape=(nfeatures,)), keras.layers.Dropout(dropout),
         keras.layers.Dense(nhidden1, activation='relu'), keras.layers.Dropout(dropout)])
    if nhidden2 != 0:
        model.add(keras.layers.Dense(nhidden2, activation='relu'))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))

    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.BinaryCrossentropy(), metrics=METRICS)
    return model


class MyNN(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.005, nfirst=1, nhidden1=10, nhidden2=0, dropout=0, output_bias=1, batch_size=100, epochs=10,
                 scale_pos_weight=1):
        # print("mynn++",end = '')
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.nfirst = nfirst
        self.nhidden1 = nhidden1
        self.nhidden2 = nhidden2
        self.dropout = dropout
        self.output_bias = output_bias
        self.scale_pos_weight = scale_pos_weight

    def fit(self, X, y,
            **fit_params):  ##{'nhidden': 40,'nfirst': 20,'epochs': 1,'drop2': 0.2,'drop1': 0.2,'batch_size': 100}
        # ipdb.set_trace()
        try:
            if X.isnull().values.any() or y.isnull().values.any():
                print("X or y contain nans")
        except:
            pass
        self.classes_ = unique_labels(y)
        if self.scale_pos_weight is not None:
            fit_params['class_weight'] = {0: 1, 1: self.scale_pos_weight}
        self.model = KerasClassifier(build_model2,
                                     **{'nfeatures': X.shape[-1], 'lr': self.lr, 'nhidden1': self.nhidden1,
                                        'nhidden2': self.nhidden2, 'nfirst': self.nfirst, \
                                        'epochs': self.epochs, 'dropout': self.dropout, 'batch_size': self.batch_size,
                                        'output_bias': self.output_bias}, verbose=0)

        fit_paramsnoevalset = fit_params.copy()
        for k in ['eval_metric', 'eval_set']:  # ,entriesToRemove:
            fit_paramsnoevalset.pop(k, None)

        if fit_params.get('eval_set') is None:
            self.history = self.model.fit(X, y, **fit_paramsnoevalset)

        else:
            self.history = self.model.fit(X, y,
                                          validation_data=(fit_params['eval_set'][0][0], fit_params['eval_set'][0][1]),
                                          **fit_paramsnoevalset)  # shoudl be ok as kerasclassifier wrapper builds new model
            if fit_params['eval_metric'] not in [m._name for m in METRICS]:
                try:
                    scorer = SCORERS[fit_params['eval_metric']]._score_func
                except:  # like minusf1
                    scorer = fit_params['eval_metric']
                self.score = scorer(fit_params['eval_set'][0][1], self.model.predict(fit_params['eval_set'][0][0]))
            else:
                self.score = self.history.history['val_' + fit_params['eval_metric']]
        #        self.inputs=self.model.model.inputs
        #        self.outputs=self.model.model.outputs
        return self.model  # history2=model.fit(dftrain[xs], dftrain['yb'],validation_data=(dftest[xs], dftest['yb']),class_weight = {0: 1, 1: 50},verbose=0,    callbacks = [reduce_lr])

    def evals_result(self):
        return {'validation_0': self.score}

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __del__(self):
        # print(" mynn--",end = '')
        tf.keras.backend.clear_session()
        gc.collect()
        if hasattr(self, 'model'):
            del self.model


#           from numba import cuda
#           cuda.select_device(0)
#           cuda.close()
def minusf1(ypred, ytrue):  # binary logistic
    ytrue = ytrue.get_label().astype(int)
    # print(f"ypred={ypred} ytrue={ytrue}")
    ypred = ypred > 0.5
    # ypred=np.where(np.max(y_predicted,axis=1)>thresh,ypredmax,0)
    f1 = f1_score(ytrue, ypred)
    f1 = -f1
    return '-f1', f1


def minusap(ypred, ytrue):  # binary logistic
    ytrue = ytrue.get_label().astype(int)
    # print(f"ypred={ypred} ytrue={ytrue}")
    # pred=ypred>0.5
    # ypred=np.where(np.max(y_predicted,axis=1)>thresh,ypredmax,0)
    ap = average_precision_score(ytrue, ypred)
    ap = -ap
    return '-ap', ap


def minusrecall(ypred, ytrue):  # binary logistic
    ytrue = ytrue.get_label().astype(int)
    ypred = ypred > 0.5
    f1 = recall_score(ytrue, ypred)
    f1 = -f1
    return '-recall', f1


def minusbacc(y_predicted, y_true):
    d = {0: 0, 1: 1}  # 0,2:1}
    ytest = y_true.get_label().astype(int)
    ypred = npa(range(y_predicted.shape[1])).take(np.argmax(y_predicted, axis=1), axis=0)
    bacc = balanced_accuracy_score(ytest, ypred)
    bacc = -bacc
    return 'bacc', bacc


# average precision , recall  f1


from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import base64
# from mlfinlab.features.fracdiff import frac_diff_ffd
from statsmodels.tsa.stattools import grangercausalitytests

# frac_diff_ffd(df1.loc[17778][['unique_addresses_all_time']], 0.5,thresh=1e-04).plot()
# df1.loc[17778][['unique_addresses_all_time']].diff(1).plot()
# from fracdiff import Fracdiff
# DF(Fracdiff(0.5,window=100,tol_memory=None,tol_coef=None, window_policy='fixed', max_window=4096).transform(df1.loc[17778][['unique_addresses_all_time']])).plot()
# ts_differencing(df1.loc[17778][['unique_addresses_all_time']],0.5,100).plot()
# df1.loc[17778][['unique_addresses_all_time']].plot()

from io import BytesIO

PLOTFLASK = True

from collections import Counter
import math
from scipy.stats import entropy


def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    log_base: float, default = e
        specifying base for calculating entropy. Default is base e.
    Returns:
    --------
    float
    """
    log_base = math.e
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy


def cramers_v(x, y,
              bias_correction=True):  # ,              nan_strategy=_REPLACE,           nan_replace_value=_DEFAULT_REPLACE_VALUE):
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    This is a symmetric coefficient: V(x,y) = V(y,x)
    Original function taken from: https://stackoverflow.com/a/46498792/5863503
    Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
        Use bias correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328.
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    Returns:
    --------
    float in the range of [0,1]
    """
    if nan_strategy == _REPLACE:
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == _DROP:
        x, y = remove_incomplete_samples(x, y)
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if bias_correction:
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        if min((kcorr - 1), (rcorr - 1)) == 0:
            warnings.warn(
                "Unable to calculate Cramer's V using bias correction. Consider using bias_correction=False",
                RuntimeWarning)
            return np.nan
        else:
            return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    else:
        return np.sqrt(phi2 / min(k - 1, r - 1))


def theil(x, y):
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-
    categorical association. This is the uncertainty of x given y: value is
    on the range of [0,1] - where 0 means y provides no information about
    x, and 1 means y provides full information about x.
    This is an asymmetric coefficient: U(x,y) != U(y,x)
    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    """
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


# cluster correl matrix
def clustercorr(corr, t=None):
    figsflask = not isnotebook()
    res = {}
    figs = []
    ifigs = []
    bigfigs = []
    #
    # corr=df.calccorr(method=method)  #  1-df.corr
    d = sch.distance.pdist(corr.values)
    L = sch.linkage(d, method="ward")  # ward

    if figsflask: img = plotstart()
    # plt.rcParams["figure.figsize"] = (20,20)
    plt.figure(figsize=(15, 15))
    sch.dendrogram(L, labels=corr.columns, orientation='right', leaf_rotation=0)
    if figsflask: bigfigs.append(plotend(img))

    if t is None:
        t = 0.5 * d.max()
    columns = [corr.columns.tolist()[i] for i in list((np.argsort(sch.fcluster(L, t, 'distance'))))]
    print(columns)
    corr2plot = corr[columns].reindex(columns, axis=0)
    #     plt.matshow(corr2plot,cmap='RdYlGn')
    #     plt.xticks(range(len(corr2plot.columns)),corr2plot.columns,rotation=90)
    #     plt.yticks(range(len(corr2plot.columns)),corr2plot.columns)
    #     plt.show()
    if figsflask: img = plotstart()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr2plot, vmin=-1, vmax=1, square=True, annot=True, cmap='RdYlGn', fmt='.1f')
    plt.show()
    if figsflask: bigfigs.append(plotend(img))

    # fig, axs = plt.subplots(1, 4,figsize=(32,16))
    # axes = list(axs.reshape(-1))
    for i, t in enumerate(np.linspace(0.01, 2 * d.max(), 4)):
        # axes[i].title.set_text("t={:.2f}".format(t))
        columns = [corr.columns.tolist()[i] for i in list((np.argsort(sch.fcluster(L, t, 'distance'))))]
        if figsflask: img = plotstart()
        plt.figure(figsize=(10, 10))
        plt.matshow(corr[columns].reindex(columns, axis=0), cmap='RdYlGn', vmin=-1, vmax=1)
        plt.title(f"t={t:.2f}")
        plt.show()
        if figsflask: figs.append(plotend(img))

    res['corrdf'] = corr2plot
    if figsflask:
        res['figs'] = figs
        res['bigfigs'] = bigfigs
    return res


pd.core.frame.DataFrame.clustercorr = clustercorr


def seasonalitydf(df, col):
    dfu = df[col].unstack().T
    dfun = (dfu / dfu.max(axis=0)).mean(
        axis=1)  # across all coins (df11/df11.max(axis=0)).mean(axis=1)#.groupby(ts.dt.day)
    dfgroup = dfun.groupby(dfun.index.weekday).mean()
    (dfgroup / dfgroup.max()).plot(kind='bar', legend=False, title=col)  # 0 = monday


def seasonalityts(
        ts):  # mon-sunday  adjustments: http://www.bgu.ac.il/~shalit/Publications/Garch_ApplFinEcon2.pdf OLS for returns and vols
    try:
        ts = ts.unstack().T
        dfun = (ts / ts.max(axis=0)).mean(
            axis=1)  # across all coins (df11/df11.max(axis=0)).mean(axis=1)#.groupby(ts.dt.day)
    except:
        dfun = (ts / ts.max(axis=0))
        pass  # not multindex

    for attr in ['weekday', 'hour', 'minute', 'second', 'day']:
        dfgroup = dfun.groupby(getattr(dfun.index, attr)).mean()
        (dfgroup / dfgroup.max()).plot(marker='o', legend=False, title=attr)  # 0 = monday
        plt.show()

    seasonal_decompose(ts, model='additive', freq=1).plot()
    plt.show()

    # plt.plot(result.seasonal)
    # print(result.resid)
    # print(result.observed)


# for col in df1.columns:#['active_addresses']:#df1.columns:
#     print(col)
#     seasonality(df1,col)

def fdiff1(series, order, lags):
    def getWeights(d, lags):
        # return the weights from the series expansion of the differencing operator
        # for real orders d and up to lags coefficients
        w = [1]
        for k in range(1, lags):
            w.append(-w[-1] * ((d - k + 1)) / k)
        w = np.array(w).reshape(-1, 1)
        return w

    # for real orders order up to lag coefficients
    weights = getWeights(order, lags)
    res = 0
    for k in range(lag):
        res += weights[k] * series.shift(k).fillna(0)  # fillna? needed?
    return res


def plotstart():
    if PLOTFLASK:
        img = BytesIO()
        plt.clf()
        return img
    return False


def plotend(img, dpi=100):
    if img is False:
        return 0
    plt.savefig(img, format='png', dpi=dpi, bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


def adf(ts):  # return 1 if stationary with 95%
    try:  # can be array, not series
        ts = ts.dropna()
    except:
        pass
    res = adfuller(ts, maxlag=1, regression='c')[1] < 0.05
    return res


def fdiff(ts, order, method='fdifflags', lags=10, thresh=1e-04):
    if method == 'fdiffthresh':  # mlfinlib
        resdf = frac_diff_ffd(DF(ts), order, thresh)
        return resdf.iloc[:, 0]
    elif method == 'fdifflags':
        return fdiff1(ts, order, lags)


pd.core.series.Series.fdiff = fdiff
pd.core.series.Series.adf = adf


def adfdf(df):
    res = DF()
    for col in df.columns:
        res[col] = [adf(df[col])]
    return res


pd.core.frame.DataFrame.adf = adfdf


def fdiffdf(df, order, method='fdifflags', lags=10, thresh=1e-04):
    res = DF()
    for col in df.columns:
        res[col] = fdiff(df[col], order=order, method=method, lags=lags, thresh=thresh)
    return res


pd.core.frame.DataFrame.fdiff = fdiffdf


def armaxgarch(ts, plot=False):
    pass


def analysets(ts):
    ts.adf()
    ts.seasonality()  # seasonal_decompose(df1.loc[17778][ 'unique_addresses_all_time'].pow(0.1), model='additive')
    ts.abnormality()  # incluen pentropy
    ts.smoothness()
    ts.nans()
    ts.nunique()
    ts.describe()


def analysedf(df, y=None, xs=None):
    for col in df.columns:
        analysets(df[col])
    # df.adf()
    df.cointegration()  # only if cointegrated same order (usually 1)  use johanses and grangerengel
    df.corrplot()
    # df.seasonality()#  seasonal_decompose(df1.loc[17778][ 'unique_addresses_all_time'].pow(0.1), model='additive')
    # df.abnormality() #incluen pentropy
    # df.smoothness()
    # df.describe()
    df.granger(y, xs)
    return


def analysedfmulti(df):  # multiindex  id ts
    df.groupby('id').analysedf()


def cointegration(df, det_order=0, k_ar_diff=1, plot=False):
    # add engel granger bivariate cointegration: statsmodels.tsa.stattools.coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag='aic', return_results=None)[source]
    # critical values depend on the trend assumptions and may not be appropriate for models that contain other deterministic regressors.
    # all ts must be integrated of same order
    df = df.dropna()
    stationarity = npa([adf(df[col]) for col in df.columns])
    rescoint = coint_johansen(df, det_order=0, k_ar_diff=1)
    trace = rescoint.lr1 > rescoint.cvt[:, 1]  # 95%   >0 >1 >2
    maxeig = rescoint.lr2 > rescoint.cvm[:, 1]  # 95%
    # rescoint.eig
    linearcomb = (np.dot(df, rescoint.evec[:, 0]))
    res = {}
    if plot:
        figs = []
        img = plotstart()
        DF(linearcomb).plot(title='1st eigenvector* all ts')
        figs.append(plotend(img))
        res['figs'] = figs
    res = {'stationary': stationarity, 'linearcombstationary': adf(linearcomb), 'trace': trace, 'maxeig': maxeig,
           'maxeigv': rescoint.evec[:, 0]}  # >0 >1 >2 >3  cointegral relations
    return res


# cointegration(df1.loc[17778][['unique_addresses_all_time','zero_balance_addresses_all_time']],plot=False) #'flask'

pd.core.frame.DataFrame.cointegration = cointegration


def plotts(y, lags=None, bins=40, figsize=(10, 8), resample='s', lowpass=0.2):
    style = 'bmh'
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        plt.tight_layout()
        plt.show()
        y.hist(bins=bins)
        plt.show()
        print(y.describe())
        print(y.index)

        # x = np.linspace(0,5,100)
    # df[['diff']].stdscale()#iloc[:20]#np.sin(2*np.pi*x)+np.sin(30*np.pi*x)

    # y = np.sin(2*np.pi*x)+np.sin(30*np.pi*x)
    try:
        y = y.resample(resample).mean().ffill()
    except Exception as e:
        print(f"can't resample{str(e)}")
    f = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), d=1)  # x[1]-x[0])
    plt.plot(freq, abs(f) ** 2)
    plt.title('FFT')
    plt.show()
    fft2 = f.copy()
    fft2[np.abs(freq) > lowpass] = 0  # 1.1
    plt.plot(np.real(fftpack.ifft(fft2)))
    plt.title(f"lowpass={lowpass}")
    plt.show()
    print('resamople.mean.ffill')
    seasonalityts(y)
    return


def plottsdf(df, bins=40):
    for col in df.columns:
        plotts(df[col], bins=bins)


pd.core.frame.DataFrame.plotts = plottsdf
pd.core.series.Series.plotts = plotts

from scipy.stats import chi2 as schi2


def smoothcoef(y):
    y = npa(y).flatten()
    ymax = max(np.abs(y).max(), 0.0000001)
    tv = np.zeros(len(y))
    for i in range(len(y) - 1):
        try:
            tv[i] = np.abs(np.diff(np.delete(y, i), 1)).sum()
        except:
            ipdb.set_trace()
    tv = np.delete(tv, [0, len(tv) - 1])
    tv = np.abs(np.diff(tv, 1))
    return tv.max() / ymax


def abnormality(ts, thresh, retpoints=False, plot=False):  # high values means abnormal
    # ddfcol=ts.shift(-1)-ts
    avg = ts.mean()
    var = ts.var()
    # print((c0t0[col]-avg)**2/var)
    nans = ts.isnull().sum() / len(ts)
    abnormal = (ts - avg) ** 2 / var > schi2.interval(1 - thresh, 1)[1]

    if plot:
        if (plot == 'abnormal' and abnormal.any()) or plot == 'all':
            plt.figure(figsize=(4, 4))
            plt.clf()
            # print(ts.index,ts.values)  if multiindex does world scatter ts.index
            plt.scatter(ts.droplevel(0).index, ts.values, c=abnormal, cmap='bwr',
                        marker='.')  # df[df.coin_id==coin_id][col+'_abnormal']
            plt.show()
    res = {}
    if retpoints:
        res['abnpoints'] = ts[abnormal]
    # df[df.coin_id==coin_id][col+'_abnormal']=df[df.coin_id==coin_id][col+'_abnormal']>abn_th
    return {**res, 'abnormal': float(abnormal.any()), 'nans': nans, 'nunique': 1 - ts.nunique() / len(ts),
            'smooth': smoothcoef(ts.dropna()), 'pentr': 1 - pentropy(ts.dropna(), normalize=True)}


def abnormalitydf(df, cols=None, thresh=0.001):
    badcoins = []
    if cols is None:
        cols = df.columns
    for id in df.index.unique(level='id'):
        for col in cols:
            thresh = thresh  # 00001
            dfcol = df.query('id==@id')[col]
            res = abnormality(dfcol, thresh)
            badcoins.append({**res, 'id': id, 'colname': col})

    return badcoins
    # c0t0[col].scatter()


def displayabnormal(df, thresh):
    for t in df:
        id = t[0]
        col = t[1]
        print("id={} col={}".format(id, col))
        abnormality(df1.query('id==@id')[col], thresh=thresh, retpoints=True, plot='all')


def dptest(df):
    pass
    # nonlinear granger lags 1,2,3,4   distance=1.5  a right-tailed version https://www.agriculturejournals.cz/publicFiles/376_2016-AGRICECON.pdf


def granger(df, maxlag=10):
    """
    Remember that Granger causality in its simplest form consists of an F-Test for the R2 of the two regressions: y=const+y[-1]+e vs. y=const+y[-1]+x[-1]+e in order to see if the R2 from the second regression is higher
    y = index   x= columns  y=ax+eps 
    if linear model is not good (if best R2 is too high, then makes no sense to to calc difference in R2)
    """
    res = DF(np.zeros((len(df.columns), len(df.columns))), columns=df.columns, index=df.columns)
    for c in range(len(df.columns)):
        for r in range(len(df.columns)):
            gc_res = grangercausalitytests(df[[df.columns[r], df.columns[c]]], maxlag, verbose=False)
            pvals = [round(gc_res[i + 1][0]['ssr_ftest'][1], 2) for i in range(maxlag)]
            res.iloc[r, c] = min(pvals)
    return res


def scales(s, method='ss'):
    """
    normalization by scaling  series
    Args:
        method 
            ss standard scale
            mm minmax scale
    """
    if method == 'ss':
        try:
            s1 = (s - s.mean()) / s.std()
        except Exception as e:
            print(f"stddscale exception {str(e)}")
    return s1


pd.core.series.Series.scale = scales


def scaledf(df, method='ss', inplace=False):
    """
    normalization by scaling  df
    Args:
        method 
            ss standard scale
            mm minmax scale
    """
    df = df if inplace else df.copy()
    for col in df.columns:
        try:
            df[col] = df[col].scale(method=method)  # (df[col]-df[col].mean())/df[col].std()
        except Exception as e:
            print(f"scale exception {str(e)}")
    return df


pd.core.frame.DataFrame.scale = scaledf


def addscale(df, cols=None, method="ss", inplace=False, retfnames=False):
    df = df if inplace else df.copy()
    if cols is None:
        cols = df.columns
    fnames = []
    for col in cols:
        colname = f'{method}.{col}'
        fnames.append(colname)
        df[colname] = df[col].scale(method=method)
    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addscale = addscale


def adddiffs(df, lags, cols=None, inplace=False, method='rl', order=0.7, thresh=1e-4, retfnames=False,
             dropna=True):  # -1 +1
    df = df if inplace else df.copy()
    if cols is None:
        cols = df.columns
    fnames = []

    if type(lags) != type([]):
        lags = [lags]

    for lag in lags:
        for col in cols:
            if method == 'ra':
                colname = 'ra' + '{:+d}.'.format(-lag) + col
                fnames.append(colname)
                df[colname] = df[col] - df[col].shift(lag)
            elif method == 'rl':
                colname = 'rl' + '{:+d}.'.format(-lag) + col
                fnames.append(colname)
                df[colname] = np.log(df[col].clip(lower=1e-7)) - np.log(df[col].shift(lag).clip(lower=1e-7))
            elif method == '':
                colname = '{:+d}.'.format(-lag) + col
                fnames.append(colname)
                df[colname] = df[col].shift(lag)
            elif method == 'pnlb':
                colname = 'pnlb{:+d}.'.format(-lag) + col
                fnames.append(colname)
                df[colname] = df[col].shift(lag) / df[col] - 1.
            elif method == 'fdifflags':
                colname = method + ',' + str(lag) + '.' + col
                fnames.append(colname)
                df[colname] = df[col].fdiff(order=order, method=method, lags=lag)
            elif method == 'fdiffthresh':
                colname = method + ',' + str(thresh) + '.' + col
                fnames.append(colname)
                df[colname] = df[col].fdiff(order=order, method=method, thresh=thresh)
    df = df.dropna() if dropna else df
    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.adddiffs = adddiffs


def addday(df, inplace=False, retfnames=False):  # 0 is monday
    df = df if inplace else df.copy()
    df['weekday'] = df.index.weekday
    df['day'] = df.index.day
    df['hour'] = df.index.hour

    fnames = ['day', 'hour', 'weekday']

    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addday = addday


def calcma(df, lag, method='ma'):
    if method == 'ma':
        res = df.rolling(lag).mean()
    elif method == 'ewm':
        res = df.ewm(span=lag).mean()
    return res


pd.core.series.Series.addma = calcma


def addma(df, lags, cols=None, inplace=False, method='ma', retfnames=False, dropna=True):  # ewm
    df = df if inplace else df.copy()
    fnames = []
    if cols is None:
        cols = df.columns

    if type(lags) != type([]):
        lags = [lags]

    for lag in lags:
        for col in cols:
            res = calcma(df[col], lag=lag, method=method)
            fname = method + str(lag) + '.' + col
            df[fname] = res
            fnames.append(fname)

    df = df.dropna() if dropna else df
    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addma = addma


def calcstd(df, lag, method='ma'):
    if method == 'ma':
        res = df.rolling(lag).std()
    elif method == 'ewm':
        res = df.ewm(span=lag).std()
    return res


pd.core.series.Series.addstd = calcstd


def addstd(df, lags, cols=None, inplace=False, method='ma', retfnames=False, dropna=True):  # ewm
    df = df if inplace else df.copy()
    fnames = []
    if cols is None:
        cols = df.columns

    if type(lags) != type([]):
        lags = [lags]

    for lag in lags:
        for col in cols:
            res = calcstd(df[col], lag=lag, method=method)
            fname = method + 'std' + str(lag) + '.' + col
            df[fname] = res
            fnames.append(fname)

    df = df.dropna() if dropna else df

    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addstd = addstd


def addbb(df, lags, cols=None, inplace=False, methodma='ewm', methodstd='ewm', nstdh=2, nstdl=2, retfnames=False,
          dropna=True):
    df = df if inplace else df.copy()
    fnames = []
    if cols is None:
        cols = df.columns

    if type(lags) != type([]):
        lags = [lags]

    for lag in lags:
        for col in cols:
            ma = df[col].addma(lag, method=methodma)
            std = df[col].addstd(lag, method=methodstd)
            fname = 'bbh.' + col
            df[fname] = ma + nstdh * std
            fnames.append(fname)
            fname = 'bbl.' + col
            df[fname] = ma - nstdl * std

    df = df.dropna() if dropna else df

    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addbb = addbb


def addbb2(df, lags, cols=None, inplace=False, methodma='ewm', methodstd='ewm', entryz=1.5, exitz=0, retfnames=False,
           dropna=True):
    df = df if inplace else df.copy()
    fnames = []
    if cols is None:
        cols = df.columns

    if type(lags) != type([]):
        lags = [lags]

    for lag in lags:
        for col in cols:
            ma = df[col].addma(lag, method=methodma)
            std = df[col].addstd(lag, method=methodstd)

            df['longentryz.' + col] = ma - entryz * std

            df['shortentryz.' + col] = ma + entryz * std

            df['longexitz.' + col] = ma + exitz * std

            df['shortexitz.' + col] = ma - exitz * std

            df['openlong.' + col] = (df[col] < df['longentryz.' + col]).astype(float)
            df['closeshort.' + col] = (df[col] < df['shortexitz.' + col]).astype(float)
            df['openshort.' + col] = (df[col] > df['shortentryz.' + col]).astype(float)
            df['closelong.' + col] = (df[col] > df['longexitz.' + col]).astype(float)

    df = df.dropna() if dropna else df

    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addbb2 = addbb2


def addrk(df, span, cols=None, inplace=False, retfnames=False, dropna=True):
    df = df if inplace else df.copy()
    fnames = []

    if cols is None:
        cols = df.columns
    for col in cols:
        res = df[col].rolling(span).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])  # pd.rank,pct=True)
        fname = 'rk' + str(span) + '.' + col
        df[fname] = res
        fnames.append(fname)

    df = df.dropna() if dropna else df
    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addrk = addrk


def getfeaturenames(fs, cols, xsraw=None):
    ys = [c for c in cols if '+' in c or c[0] == 'y']

    # filter only xsraw
    if xsraw is not None:
        cols = [col for col in cols if col.split('.')[-1] in xsraw]

    if fs == 'y':
        return ys
    if fs == 'raw':
        return list(set([c for c in cols if not '.' in c]) - set(ys))
    if fs == 'q3':
        q = []
        for i in range(2, 9):
            q.extend([c for c in cols if 'qrandn' + str(i) in c])
        return list(set(q) - set(ys))
    if fs == 'q95':
        return list(set([c for c in cols if 'qrandn0' in c]) - set(ys))
    if fs == 'all':
        return list(set(cols) - set(ys))
    if fs == 'interact':
        q = []
        for n in ['prd', 'sum', 'sub']:
            q.extend([c for c in cols if c[:3] == n])
        return list(set(q) - set(ys))


def addinteract(df, cols=None, inplace=False, ops=None, retfnames=False, dropna=True):
    df = df if inplace else df.copy()
    fnames = []
    if cols is None:
        cols = df.columns
    if ops is None:
        ops = ['sum', 'sub', 'prd']

    def sum(a, b):
        return a + b

    def sub(a, b):
        return a - b

    def prd(a, b):
        return a * b

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            for op in ops:
                try:
                    fname = op + '(' + cols[i] + ',' + cols[j] + ')'
                    df[fname] = eval(op)(df[cols[i]], df[cols[j]])
                    fnames.append(fname)
                except Exception as e:
                    print(e)

    df = df.dropna() if dropna else df
    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addinteract = addinteract


#     for col in list(set(res.columns)-set(res.columns[res.columns.str.contains('rk.')])-set(res.columns[res.columns.str.contains('q')])):
#         if col.count('.')<2:
#             res['rk.'+col]=res.groupby(res.index)[col].rank(pct=True)


def qma(s, nq, lag):
    ma = s.rolling(lag).mean()
    mstd = s.rolling(lag).std()

    res = pd.Series(np.zeros(len(s)), index=s.index)
    res[s > ma + nq * mstd] = 1
    res[s < ma - nq * mstd] = -1
    return res


def qserie(s, nq, lag, method='randn', fillnans=0):
    """
    will fill 0 to nans (when constant number repeats)

    """

    dftemp = s.rolling(lag)
    kwargs = {'raw': False}
    iloc = -1
    if type(nq) == type([]):
        norm = (len(nq) - 1) // 2
    else:
        norm = (nq - 1) // 2

    if method == 'ma':
        res = qma(s, nq, lag)

    if method == 'qcut':
        res = dftemp.apply(lambda x: pd.qcut(x, nq, labels=False, duplicates='drop').iloc[iloc] - norm,
                           **kwargs)  # pd.rank,pct=True)
    elif method == 'rank':
        res = dftemp.apply(
            lambda x: pd.qcut(x.rank(method='first'), nq, labels=False, duplicates='drop').iloc[iloc] - norm,
            **kwargs)  # pd.rank,pct=True)
    elif method == 'randn':
        res = dftemp.apply(lambda x: pd.qcut(x + 0.0000000001 * min(abs(x)) * np.random.randn(len(x)), nq, labels=False,
                                             duplicates='drop').iloc[iloc] - norm, **kwargs)  # pd.rank,pct=True)
    # logging.info(f"fillnanas={fillnans} {type(fillnans)}  {type(fillnans)==type(str)}")
    if type(fillnans) == str:
        return eval('res.' + fillnans)
    return res.fillna(fillnans)


from multiprocessing import Pool


def addqs(df, nq, lags, cols=None, inplace=False, method='randn', retfnames=False, dropna=True, fillnans=0):
    # if method=='qcut , possible that 0,5%,95%,100%  quantiles a lot of -1 , 1 instead of alot 0, randn should compensate on average
    df = df if inplace else df.copy()
    fnames = []
    if cols is None:
        cols = df.columns

    if type(lags) != type([]):
        lags = [lags]

    args = [(df[col], nq, lag, method, fillnans) for lag in lags for col in cols]
    fnames = ['q' + method + str(nq).replace("[", "").replace("]", "").replace(" ", "") + ',' + str(lag) + '.' + col for
              lag in lags for col in cols]
    with Pool() as pool:
        res = pool.starmap(qserie, args)

    for i in range(len(res)):
        df[fnames[i]] = res[i]
    df = df.dropna() if dropna else df

    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addqs = addqs


def addqs2(df, nq, lags, cols=None, inplace=False, method='randn', retfnames=False, dropna=True):
    # if method=='qcut , possible that 0,5%,95%,100%  quantiles a lot of -1 , 1 instead of alot 0, randn should compensate on average
    df = df if inplace else df.copy()
    fnames = []
    if cols is None:
        cols = df.columns

    if type(lags) != type([]):
        lags = [lags]

    for lag in lags:

        for col in cols:
            # if lag is None:
            #     dftemp=df[[col]]
            #     kwargs={}
            #     iloc=list(range(len(dftemp)))
            # else:
            dftemp = df[col].rolling(lag)
            kwargs = {'raw': False}
            iloc = -1
            if type(nq) == type([]):
                norm = (len(nq) - 1) // 2
            else:
                norm = (nq - 1) // 2
            if method == 'qcut':
                res = dftemp.apply(lambda x: pd.qcut(x, nq, labels=False, duplicates='drop').iloc[iloc] - norm,
                                   **kwargs)  # pd.rank,pct=True)
            elif method == 'rank':
                res = dftemp.apply(
                    lambda x: pd.qcut(x.rank(method='first'), nq, labels=False, duplicates='drop').iloc[iloc] - norm,
                    **kwargs)  # pd.rank,pct=True)
            elif method == 'randn':
                res = dftemp.apply(lambda x: pd.qcut(x + 0.0000000001 * np.random.randn(len(x)), nq, labels=False,
                                                     duplicates='drop').iloc[iloc] - norm,
                                   **kwargs)  # pd.rank,pct=True)
            fname = 'q' + method + str(nq).replace("[", "").replace("]", "").replace(" ", "") + ',' + str(
                lag) + '.' + col
            df[fname] = res
            fnames.append(fname)

    df = df.dropna() if dropna else df

    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addqs2 = addqs2


# to generate feature such that sum is product/division
def addlog(df, cols=None, inplace=False, retfnames=False, dropna=True):
    df = df if inplace else df.copy()
    fnames = []
    if cols is None:
        cols = df.columns
    for col in cols:
        fname = 'l.' + col
        try:
            df[fname] = np.log(0.000001 + df[col].abs())
            fnames.append(fname)
        except Exception as e:
            print(e)
            pass

    df = df.dropna() if dropna else df
    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addlog = addlog


def addpnl(df, periods=None, askcol='ap', bidcol='bp', inplace=False, fee=0.0004, retfnames=False, dropna=True):
    df = df if inplace else df.copy()
    fnames = []
    if periods is None:
        periods = [1]
    for k in periods:
        sk = str(k)
        fname = 'pnlb' + '+' + sk
        df[fname] = (df[bidcol].shift(-1) / df[askcol] - 1) - fee
        fnames.append(fname)
        fname = 'pnls' + '+' + sk
        df[fname] = -(df[askcol].shift(-1) / df[bidcol] - 1) - fee
        fnames.append(fname)

    df = df.dropna() if dropna else df

    if inplace and retfnames:
        return fnames
    if not inplace and retfnames:
        return df, fnames
    if not inplace and not retfnames:
        return df


pd.core.frame.DataFrame.addpnl = addpnl

if __name__ == '__main__':

    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4, 5, 6])

    tscv = TimeSeriesSplit(gap=1, max_train_size=2, n_splits=2, test_size=1)
    print(tscv)
    print(showsplit(DF(X), tscv).myround2())
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(random_state=0)
    print(BorutaPy(clf).fit(np.array([1, 2, 3, 4]).reshape(-1, 1), np.array([1, 2, 3, 4])))

    unittestmodelrf = RandomForestClassifier(n_estimators=2, max_depth=2)
    unittestmodelxgb = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=1, learning_rate=1, max_depth=10,
                                         alpha=1, n_estimators=5)
    unittestmodelnn = MyNN(lr=0.1, **{'scale_pos_weight': 1, 'output_bias': 0, 'nhidden1': 1, 'nfirst': 1, 'epochs': 2,
                                      'dropout': 0, 'batch_size': 2})
    testx1 = npa([-1, -2, -3, -4, -5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    np.random.shuffle(testx1)
    testx2 = testx1 + np.random.randn(len(testx1))
    testx3 = np.random.randn(len(testx1))
    testy = (testx1 > 0).astype(int)
    unittestxs = ['x1', 'x2', 'x3', 'q.x3']
    dfunittest = DF({'x1': testx1, 'x2': testx2, 'x3': testx3, 'q.x3': pd.qcut(testx3, 3, labels=False), 'y': testy})
    print(fs1(dfunittest, 'y', unittestxs))
    runselector(dfunittest.iloc[:10], y='y', xs=unittestxs, model=unittestmodelxgb, nansy='.fillna(0)', nansx=None,
                verbose=2, methods=['sfsb', 'sfsf', 'eli', 'shap'], dftest=dfunittest.iloc[-10:], scoring='f1',
                eval_metric='auc', cv=2)
    runselector(dfunittest.iloc[:10], y='y', xs=unittestxs, model=unittestmodelnn, nansy='.fillna(0)', nansx=None,
                verbose=2, methods=['sfsb', 'sfsf', 'eli', 'shap'], dftest=dfunittest.iloc[-10:], scoring='f1',
                eval_metric='auc', cv=2)
