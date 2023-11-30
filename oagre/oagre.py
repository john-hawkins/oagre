import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor, is_classifier
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.exceptions import NotFittedError


class OAGRE(BaseEstimator, RegressorMixin):
    """
    OAGRE : Outlier Attenuated Gradient-Boosted Regressor
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
    >>> ogre = OAGRE(
    ...    classifier=DecisionTreeClassifier(max_depth=5, random_state=0),
    ...    regressor=DecisionTreeRegressor(max_depth=5, random_state=0)
    ... )
    >>> ogre.fit(X, y)
    >>> ogre.predict(X)[:5]
    """

    #####################################################################
    def __init__(self, classifier, regressor, lr=0.1) -> None:
        """Initialize the meta-model with base models."""
        self.classifier = classifier
        self.regressor = regressor
        self.lr = lr
        self.n_estimators = 20
        self.class_threshold = 0.5

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
        OAGRE
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
        # DEBUGGING PRINT STATEMENTS - TODO ADD A LOGGING FLAG
        #print("Base Model Mean Absolute Error: ", np.absolute(errors).mean() )
        #print("           Max Absolute Error: ", np.absolute(errors).max() )
        preds_buffer = preds
        self.threshold = 3
        self.depth_ = 0
        self.classifiers_ = []
        self.regressors_ = []
        self.gamma_ = []
        # If the base model has no residual error then we stop processing
        if np.absolute(errors).sum() == 0.0:
            process = False 
        else:
            process = True

        while process:
            mu_ = np.mean(errors)
            sigma_ = np.std(errors)
            targs = np.ones(len(y))
            # #########################################################
            # We are assuming that the error is centred on zero
            # TODO: Implement tests and alternative approaches
            upper = 0.0 + self.threshold * sigma_
            lower = 0.0 - self.threshold * sigma_
            targs[errors>upper] = 0
            targs[errors<lower] = 0
            if targs.mean()==1.0:
                maxerr = errors.max()
                minerr = errors.min()
                targs[errors==maxerr] = 0
                if minerr < 0:
                    targs[errors==minerr] = 0
            self.classifiers_.append( clone(self.classifier) )            
            self.classifiers_[self.depth_].fit(X, targs, sample_weight)
            try:
                temp1 = self.classifiers_[self.depth_].predict_proba(X)
                if temp1.shape[1]>1:
                    temp = temp1[:,1]
                else:
                    temp = temp1[:,0]
            except:
                print("ERROR in OAGRE: ")
                print(" Targets length", str(len(targs)), " mean value", str(targs.mean()) )
                print(self.classifiers_)
                print(temp1)
                print(temp.shape)
            # ########################################################################
            # We have experimented with thresholding the probabilities
            # TODO: Add this as an optional flag
            #temp = np.where(temp>self.class_threshold, 1.0, 0.0)
            # ####################################################################################
            # Now extract just the records with error within bounds to train the residual regression model
            y_temp = errors[targs==1]
            X_temp = X[targs==1]
            #print("OAGRE Excluded Data:", str(len(errors[targs==0])))
            self.regressors_.append( clone(self.regressor) )
            self.regressors_[self.depth_].fit(X_temp, y_temp, sample_weight)
            temp2 = self.regressors_[self.depth_].predict(X)
            mypreds = temp * temp2
            def fit_gamma(x):
                temp1 = preds_buffer - x * self.lr * mypreds
                temp2 = temp1 - y
                return (temp2*temp2).mean()
            rez = optimize.minimize_scalar(fit_gamma)
            if rez.success:
                self.gamma_.append(rez.x)
            else:
                self.gamma_.append(1)
            # WE HAVE REMOVED THE USE OF THE LEARNING RATE FOR MORE TESTING           
            # current_preds = preds_buffer - self.lr * self.gamma_[self.depth_] * mypreds            
            current_preds = preds_buffer - mypreds
            preds_buffer = current_preds
            errors = preds_buffer - y
            self.depth_ = self.depth_ + 1
            self.threshold = 3 - (self.depth_/self.n_estimators)
            if self.depth_ == self.n_estimators:
                process = False
            if np.absolute(errors).sum() == 0.0:
                process = False                  # If there is no residual, then we stop processing

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
        # DEBUGGING PRINT STATEMENTS
        #print("Mean base pred:", preds.mean())
        index = 0
        while index < self.depth_:
            temp1 = self.classifiers_[index].predict_proba(X)
            if temp1.shape[1]>1:
                temp = temp1[:,1]
            else:
                temp = temp1[:,0]
            # THRESHOLDING EXPERIMENT
            #temp = np.where(temp>self.class_threshold, 1.0, 0.0)
            temp2 = self.regressors_[index].predict(X)
            mypreds = temp * temp2
            adjustment = mypreds # * self.lr * self.gamma_[index]
            newpreds = preds - adjustment
            # DEBUGGING PRINT STATEMENTS
            # print("Proportion of records to adjust:", temp.mean())
            # print("Mean adjustment:", adjustment.mean())
            # print("Min adjustment:", adjustment.min())
            # print("Max adjustment:", adjustment.max())
            # print("After depth", index, " preds:", newpreds.mean())
            preds = newpreds
            index = index + 1

        return preds

