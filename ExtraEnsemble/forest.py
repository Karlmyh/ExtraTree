import numpy as np
from ExtraTree import StandardTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from multiprocessing import Pool


BASE_LEARNER = {
    "standard_tree_regressor": StandardTreeRegressor,
    "extra_tree_regressor": ExtraTreeRegressor
    }

def train_parallel(input_tuple):
    tree, X, y, random_state, max_samples = input_tuple
    np.random.seed(random_state)
    bootstrap_idx = np.random.choice(X.shape[0], int(np.ceil(X.shape[0] * max_samples)))
    return tree.fit( X[bootstrap_idx], y[bootstrap_idx])

def pred_parallel(input_tuple):
    tree, X = input_tuple
    return tree.predict(X)


class BaseForest(object):
    def __init__(self,  n_estimators = 20, 
                 max_features = 1.0, 
                 max_samples = 1.0,
                 ensemble_parallel = 0,
                 splitter = "maxedge", 
                 base_learner = "standard_tree_regressor",
                 min_samples_split = 5, 
                 min_samples_leaf = 2,
                 max_depth = 2, 
                 order = 0, 
                 log_Xrange = True, 
                 random_state = 666,
                 parallel_jobs = 0, 
                 V = 2,
                 r_range_low = 0,
                 r_range_up = 1,
                 lamda = 0.01, 
                 search_number = 10,
                 threshold = 0
                 ):
        
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.splitter = splitter
        self.base_learner = base_learner
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.order=order
        self.log_Xrange = log_Xrange
        self.random_state = random_state
        self.parallel_jobs = parallel_jobs
        self.V = V
        self.r_range_up =r_range_up
        self.r_range_low =r_range_low
        self.lamda=lamda
        self.search_number = search_number
        self.threshold = threshold
        self.ensemble_parallel = ensemble_parallel

        self.trees = []

        
    def fit(self, X, y):
        
        assert self.ensemble_parallel * self.parallel_jobs == 0
        
        if self.ensemble_parallel != 0:
            for i in range(self.n_estimators):
                self.trees.append(BASE_LEARNER[self.base_learner](splitter = self.splitter, 
                                                 min_samples_split = self.min_samples_split,
                                                 min_samples_leaf = self.min_samples_leaf,
                                                 max_depth = self.max_depth,
                                                 order = self.order, 
                                                 log_Xrange = self.log_Xrange, 
                                                 random_state = i,
                                                 parallel_jobs = self.parallel_jobs,
                                                 V = self.V,
                                                 r_range_low = self.r_range_low,
                                                 r_range_up = self.r_range_up,
                                                 lamda = self.lamda,
                                                 max_features = self.max_features,
                                                 search_number = self.search_number,
                                                 threshold = self.threshold))

            with Pool( min(self.ensemble_parallel, self.n_estimators)) as p:
                self.trees = p.map(train_parallel, [(self.trees[i],X,y,i,self.max_samples) for i in range(self.n_estimators)])
                
                
        else:
            for i in range(self.n_estimators):
                np.random.seed(i)
            
                bootstrap_idx = np.random.choice(X.shape[0], int(np.ceil(X.shape[0] * self.max_samples)))



                self.trees.append(BASE_LEARNER[self.base_learner](splitter = self.splitter, 
                                                 min_samples_split = self.min_samples_split,
                                                 min_samples_leaf = self.min_samples_leaf,
                                                 max_depth = self.max_depth,
                                                 order = self.order, 
                                                 log_Xrange = self.log_Xrange, 
                                                 random_state = i,
                                                 parallel_jobs = self.parallel_jobs,
                                                 V = self.V,
                                                 r_range_low = self.r_range_low,
                                                 r_range_up = self.r_range_up,
                                                 lamda = self.lamda,
                                                 max_features = self.max_features,
                                                 search_number = self.search_number,
                                                 threshold = self.threshold))

                self.trees[i].fit(X[bootstrap_idx] , y[bootstrap_idx])
        
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in [ "n_estimators" ,'min_samples_split', "max_features", "max_samples"
                    "splitter", "min_samples_leaf", "max_depth", "order", "V", 
                    "r_range_low", "r_range_up", "lamda",
                    "search_number", "threshold"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
    
    def predict(self, X):
        if self.ensemble_parallel == 0:
            y_hat = np.zeros(X.shape[0])
            for i in range(self.n_estimators):
                y_hat +=  self.trees[i].predict(X)
            y_hat /= self.n_estimators
            return y_hat
        else:
            with Pool(min(self.ensemble_parallel, self.n_estimators)) as pp:
                y_hat = pp.map(pred_parallel, [( self.trees[i], X) for i in range(self.n_estimators)])
            y_hat = np.array(y_hat).mean(axis = 0)

            return y_hat
    
    
class StandardForestRegressor(BaseForest):
    def __init__(self, n_estimators = 20, 
                 max_features = 1.0, 
                 max_samples = 1.0,
                 ensemble_parallel = 0,
                 splitter = "maxedge", 
                 min_samples_split = 5, 
                 min_samples_leaf = 2,
                 max_depth = 2, 
                 order = 0, 
                 log_Xrange = True, 
                 random_state = 666,
                 parallel_jobs = 0, 
                 V = 2,
                 r_range_low = 0,
                 r_range_up = 1,
                 lamda = 0.01, 
                 search_number = 10,
                 threshold = 0):
        super(StandardForestRegressor, self).__init__(n_estimators = n_estimators,
                                                 max_features = max_features,
                                                 max_samples = max_samples,
                                                 ensemble_parallel = ensemble_parallel,
                                                 splitter = splitter,
                                                 base_learner = "standard_tree_regressor", 
                                                 min_samples_split = min_samples_split,
                                                 min_samples_leaf = min_samples_leaf,
                                                 max_depth = max_depth, 
                                                 order = order,
                                                 log_Xrange = log_Xrange, 
                                                 random_state = random_state,
                                                 parallel_jobs = parallel_jobs,
                                                 V = V,
                                                 r_range_low = r_range_low,
                                                 r_range_up = r_range_up,
                                                 lamda = lamda,
                                                 search_number = search_number,
                                                 threshold = threshold)
        
    def score(self, X, y):
        return -MSE(self.predict(X),y)
    
    
    
    
class ExtraForestRegressor(BaseForest):
    def __init__(self, n_estimators = 20, 
                 max_features = 1.0, 
                 max_samples = 1.0,
                 ensemble_parallel = 0,
                 splitter = "maxedge", 
                 min_samples_split = 5, 
                 min_samples_leaf = 2,
                 max_depth = 2, 
                 order = 0, 
                 log_Xrange = True, 
                 random_state = 666,
                 parallel_jobs = 0, 
                 V = 2,
                 r_range_low = 0,
                 r_range_up = 1,
                 lamda = 0.01, 
                 search_number = 10,
                 threshold = 0):
        super(ExtraForestRegressor, self).__init__(n_estimators = n_estimators,
                                                 max_features = max_features,
                                                 max_samples = max_samples,
                                                 ensemble_parallel = ensemble_parallel,
                                                 splitter = splitter,
                                                 base_learner = "extra_tree_regressor", 
                                                 min_samples_split = min_samples_split,
                                                 min_samples_leaf = min_samples_leaf,
                                                 max_depth = max_depth, 
                                                 order = order,
                                                 log_Xrange = log_Xrange, 
                                                 random_state = random_state,
                                                 parallel_jobs = parallel_jobs,
                                                 V = V,
                                                 r_range_low = r_range_low,
                                                 r_range_up = r_range_up,
                                                 lamda = lamda,
                                                 search_number = search_number,
                                                 threshold = threshold)
        
    def score(self, X, y):
        return -MSE(self.predict(X),y)