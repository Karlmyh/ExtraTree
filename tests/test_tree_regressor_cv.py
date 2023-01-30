import numpy as np 
from ExtraTree import StandardTreeRegressor, ExtraTreeRegressor
from sklearn.model_selection import GridSearchCV



def test_standard_tree_regressor():
    
                
    np.random.seed(666)
    X_train = np.random.rand(200).reshape(-1,2)
    X_test = np.random.rand(200).reshape(-1,2)
    Y_train = np.ones(100)

    parameters = {"max_depth":[2, 4, 6, 8],
                  "splitter":["purely", "midpoint", "maxedge", "msereduction", "msemaxedge"],
                  "threshold":[0, 0.01],
                  }
    
    cv_model_standard_tree = GridSearchCV(estimator = StandardTreeRegressor(), param_grid = parameters, cv = 3) 
    cv_model_standard_tree.fit(X_train, Y_train)
    
    model_standard_tree = cv_model_standard_tree.best_estimator_
    assert (model_standard_tree.predict(X_test)==1).all()
    
    
def test_extra_tree_regressor():
                    
    np.random.seed(666)
    X_train = np.random.rand(200).reshape(-1,2)
    X_test = np.random.rand(200).reshape(-1,2)
    Y_train = np.ones(100)

    parameters = {"max_depth":[2, 4, 6, 8],
                  "splitter":["purely", "midpoint", "maxedge", "msereduction", "msemaxedge"],
                  "threshold":[0, 0.01],
                  "order":[0, 1, 5],
                  "lamda":[0.001]
                  }
    
    cv_model_extra_tree = GridSearchCV(estimator = ExtraTreeRegressor(), param_grid = parameters, cv = 3) 
    cv_model_extra_tree.fit(X_train, Y_train)
    
    model_extra_tree = cv_model_extra_tree.best_estimator_
    assert ((model_extra_tree.predict(X_test)-1)**2).mean()<0.01
    
    
    



def test_standard_tree_regressor_noise():
    
                
    np.random.seed(666)
    X_train = np.random.rand(200).reshape(-1,2)
    X_test = np.random.rand(200).reshape(-1,2)
    Y_train = np.ones(100) + np.random.normal(scale = 0.1, size = 100)

    parameters = {"max_depth":[2, 4, 6, 8],
                  "splitter":["purely", "midpoint", "maxedge", "msereduction", "msemaxedge"],
                  "threshold":[0, 0.01],
                  }
    
    cv_model_standard_tree = GridSearchCV(estimator = StandardTreeRegressor(), param_grid = parameters, cv = 3) 
    cv_model_standard_tree.fit(X_train, Y_train)
    
    model_standard_tree = cv_model_standard_tree.best_estimator_
    assert (((model_standard_tree.predict(X_test)-1)**2).mean() < 0.15).all()
    
    
def test_extra_tree_regressor_noise():
                    
    np.random.seed(666)
    X_train = np.random.rand(200).reshape(-1,2)
    X_test = np.random.rand(200).reshape(-1,2)
    Y_train = np.ones(100) + np.random.normal(scale = 0.1, size = 100)

    parameters = {"max_depth":[2, 4, 6, 8],
                  "splitter":["purely", "midpoint", "maxedge", "msereduction", "msemaxedge"],
                  "threshold":[0, 0.01],
                  "order":[0, 1, 5],
                  "lamda":[0.001]
                  }
    
    cv_model_extra_tree = GridSearchCV(estimator = ExtraTreeRegressor(), param_grid = parameters, cv = 3) 
    cv_model_extra_tree.fit(X_train, Y_train)
    
    model_extra_tree = cv_model_extra_tree.best_estimator_
    assert (((model_extra_tree.predict(X_test)-1)**2).mean() < 0.15).all()

                      
                        
                    
                    


                      
                        
                    
                    
