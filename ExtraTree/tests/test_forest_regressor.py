import numpy as np 
from ExtraEnsemble import StandardForestRegressor, ExtraForestRegressor


def test_standard_forest_regressor():
    
    for splitter in ["purely", "midpoint", "maxedge", "msereduction", "msemaxedge"]:
        for threshold in [0, 0.01]:
            for parallel_jobs in [0, 5]:
                for max_features in [0.9, 1]:
                    for max_samples in [0.9, 1]:
                
                        np.random.seed(666)
                        X_train = np.random.rand(200).reshape(-1,2)
                        X_test = np.random.rand(200).reshape(-1,2)
                        Y_train = np.ones(100)
        
                        model = StandardForestRegressor( n_estimators = 20,
                                                max_features = max_features,
                                                max_samples = max_samples,
                                                ensemble_parallel = int(5-parallel_jobs),
                                                splitter = splitter,
                                                min_samples_split = 5, 
                                                min_samples_leaf = 2,
                                                max_depth = 2, 
                                                log_Xrange = True, 
                                                random_state = 666,
                                                parallel_jobs = parallel_jobs, 
                                                search_number = 10,
                                                threshold = threshold)
                        model.fit(X_train, Y_train)
                        assert ((model.predict(X_test)-1)**2).mean()<0.03
    
    
def test_extra_forest_regressor():
    
    for splitter in ["purely", "midpoint", "maxedge", "msereduction", "msemaxedge"]:
        for order in [0,1]:
            for threshold in [0,0.01]:
                for parallel_jobs in [0,5]:
                    for lamda in [0.0001]:
                        for max_features in [0.9, 1]:
                            for max_samples in [0.9, 1]:
                    
                                np.random.seed(666)
                                X_train = np.random.rand(200).reshape(-1,2)
                                X_test = np.random.rand(200).reshape(-1,2)
                                Y_train = np.ones(100)
        
                                model = ExtraForestRegressor( n_estimators = 20,
                                                        max_features = max_features,
                                                        max_samples = max_samples,
                                                        ensemble_parallel = int(5-parallel_jobs),
                                                        splitter = splitter,
                                                        min_samples_split = 5, 
                                                        min_samples_leaf = 2,
                                                        max_depth = 2, 
                                                        order = order, 
                                                        log_Xrange = True, 
                                                        random_state = 666,
                                                        parallel_jobs = parallel_jobs, 
                                                        V = 10,
                                                        r_range_low = 0,
                                                        r_range_up = 1,
                                                        lamda = lamda, 
                                                        search_number = 10,
                                                        threshold = threshold)
                                model.fit(X_train, Y_train)
        
                              
                                assert ((model.predict(X_test)-1)**2).mean()<0.03
                    
                    
