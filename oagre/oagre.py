import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor, is_classifier
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.exceptions import NotFittedError


class OaGRe(BaseEstimator, RegressorMixin):
    """
    OaGRe : Outlier adjusted Gradient-Boosted Regressor
    A meta regressor for building regression models.
    Like standard GBM the ensemble is constructed by iteratively predicting and
    correcting the residuals. 
    ----------
    classifier : Any, scikit-learn classifier
        A classifier that answers the question:
         "Are the remaining residuals predictable or due to outliers?".
    regressor : Any, scikit-learn regressor
        A base regressor for generating each layer of the GBM by
        predicting the target or the residuals.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    >>> np.random.seed(0)
    >>> ogre = OaGRe(
    ...    classifier=DecisionTreeClassifier(random_state=0),
    ...    regressor=DecisionTreeRegressor(random_state=0)
    ... )
    >>> ogre.fit(X, y)
    OaGRe(classifier=DecisionTreeClassifier(random_state=0),
                          regressor=DecisionTreeRegressor(random_state=0))
    >>> ogre.predict(X)[:5]
    """

    #####################################################################
    def __init__(self, classifier, regressor, lr=0.1) -> None:
        """Initialize the meta-model with base models."""
        self.classifier = classifier
        self.regressor = regressor
        self.lr = lr
        self.n_estimators = 100

    #####################################################################
    def fit(self, X, y, sample_weight=None):
        """
        Fit the model.
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.
        y : np.ndarray, 1-dimensional
            The target values.
        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.
        Returns
        -------
        OaGRe
            Fitted regressor.
        Raises
        ------
        ValueError
            If `classifier` is not a classifier or `regressor` is not a regressor.
        """
        X, y = check_X_y(X, y)
        self._check_n_features(X, reset=True)
        if not is_classifier(self.classifier):
            raise ValueError(
                f"`classifier` has to be a classifier. Instance of {type(self.classifier)} received.")
        if not is_regressor(self.regressor):
            raise ValueError(f"`regressor` has to be a regressor. Instance of {type(self.regressor)} received.")

        # Train the base regressor
        self.base_regressor_ = clone(self.regressor)
        self.base_regressor_.fit( X, y, sample_weight=sample_weight)

        preds = self.base_regressor_.predict(X)
        errors = preds - y
        preds_buffer = preds
        self.threshold = 4
        self.depth_ = 0
        self.classifiers_ = []
        self.regressors_ = []
        self.gamma_ = []
        process = True

        while process:
            mu_ = np.mean(errors)
            sigma_ = np.std(errors)
            targs = np.ones(len(y))
            upper = mu_ + self.threshold * sigma_
            lower = mu_ - self.threshold * sigma_
            targs[errors>upper] = 0
            targs[errors<lower] = 0
            self.classifiers_.append( clone(self.classifier) )            
            self.classifiers_[self.depth_].fit(X, targs, sample_weight)
            temp = self.classifiers_[self.depth_].predict_proba(X)[:,1]

            # Now extract just the records within the bounds to train the next regression model
            y_temp = errors[targs==1]
            X_temp = X[targs==1]
            self.regressors_.append( clone(self.regressor) )
            self.regressors_[self.depth_].fit(X_temp, y_temp, sample_weight)
            temp2 = self.regressors_[self.depth_].predict(X)
            mypreds = temp * temp2
            def fit_gamma(x):
                temp1 = preds_buffer - x * mypreds
                temp2 = temp1 - y
                return abs(temp2).mean()
            rez = optimize.minimize_scalar(fit_gamma)
            if rez.success:
                self.gamma_[self.depth_] = rez.x
            else:
                self.gamma_[self.depth_] = 1
            current_preds = preds_buffer - self.lr * self.gamma_[self.depth_] * mypreds            
            preds_buffer = current_preds
            errors = preds_buffer - y
            self.depth_ = self.depth_ + 1
            self.threshold = 4 - (2 * self.depth_/self.n_estimators)
            if self.depth_ == self.n_estimators:
                process = False

        return self

 
    #####################################################################
    def predict(self, X):
        """
        Get predictions.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples to get predictions of.
        Returns
        -------
        y : np.ndarray, shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        preds = self.base_regressor_.predict(X)
        index = 0
        while index < self.depth_:
            temp = self.classifiers_[index].predict_proba(X)[:,1]
            temp2 = self.regressors_[index].predict(X)
            mypreds = temp * temp2
            preds = preds - self.lr * mypreds
            index = index + 1

        return preds

