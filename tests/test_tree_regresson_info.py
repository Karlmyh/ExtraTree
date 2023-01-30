import numpy as np 
from ExtraTree import ExtraTreeRegressor



    
    
def test_extra_tree_regressor_info():
    
    for splitter in ["purely", "midpoint", "maxedge", "msereduction", "msemaxedge"]:
        for order in [0,1,5]:
            for threshold in [0,0.1]:
                for parallel_jobs in [0,5]:
                    for lamda in [0,0.001]:
                    
                        np.random.seed(666)
                        X_train = np.random.rand(200).reshape(-1,2)
                        X_test = np.random.rand(2).reshape(-1,2)
                        Y_train = np.ones(100)

                        model = ExtraTreeRegressor( splitter = splitter,
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
                                                max_features = 1.0,
                                                search_number = 10,
                                                threshold = threshold)
                        model.fit(X_train, Y_train)

                        pred_weights, all_r, all_y_hat, used_r, used_y_hat = model.get_info(X_test)
                        
                        
                        assert np.abs( pred_weights[0, 0] - 1 ) < 1
                        assert all_r.shape == all_y_hat.shape
                        assert used_r.shape == used_y_hat.shape
                        assert used_r.shape[0] <= 10
                    
                    
