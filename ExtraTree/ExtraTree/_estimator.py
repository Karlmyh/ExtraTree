import numpy as np
from numba import njit



@njit
def extrapolate_regression(dt_X, dt_Y, X_extra, X_range, order, 
                              r_range_low, r_range_up, V, lamda):
    """Compute the extrapolated result based on training samples in the cell.

    Parameters
    ----------
    dt_X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_extra : array-like of shape (dim_, )
        Extrapolated points.
    
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    order : int > 0
        Extrapolation order.
    
    r_range_low : float in [0, 1]
        Lower bound of homothetic ratio to consider.
    
    r_range_up : float in [0, 1], > r_range_low
        Upper bound of homothetic ratio to consider.
        
    V : int
        Parameter for homothetic estimation. The estimations are taken at 
        i/V, i = 1, ..., V.
    
    lamda : float, >0
        Ridge regularization parameter. 
    
    Returns
    -------
    weights : array-like of shape (order, 1)
        Learnt coefficients of polynomials, weights[0,0] is the final extrapolated 
        estimation. 
        
    all_ratio : array-like of shape (n_sample_, )
        The sorted vector of homothetic ratios associated to all points in 
        the cell. 
        
    all_y_hat : array-like of shape (n_sample_, )
        The sorted vector of homothetic estimations associated to all points in 
        the cell. 
    
    all_ratio : array-like of shape (V, )
        The sorted vector of selected homothetic ratios. 
        
    all_y_hat : array-like of shape (V, )
        The sorted vector of selected homothetic estimations. 

    """
    n_pts = dt_X.shape[0]
    ratio_vec = np.zeros(n_pts)
    
    
    # standardize the cell to compute homothetic ratios
    for idx_X, X in enumerate(dt_X):
        centralized = X - X_extra
        for d in range(X_extra.shape[0]):
            positive_len = X_range[1,d] - X_extra[d]
            negative_len = X_extra[d] - X_range[0,d]
            
            if centralized[d] > 0:
                centralized[d] /= positive_len
            elif centralized[d] < 0:
                centralized[d] /= negative_len
         
        
        ratio_X = np.abs(centralized).max() 
        ratio_vec[idx_X] = ratio_X

    idx_sorted_by_ratio = np.argsort(ratio_vec)  
    
    #### all sorted ratio
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    all_ratio = sorted_ratio.ravel()
    
    #### all sorted y
    sorted_y = dt_Y[idx_sorted_by_ratio]
    sorted_y_hat = np.zeros((n_pts,1))
    for k in range(n_pts):
        sorted_y_hat[k,0] = np.mean(sorted_y[ :(k+1)])
    all_y_hat = sorted_y_hat.ravel()
    
    # compute the homothetic estimations for selected sequence
    index_by_r = np.zeros(V)
    for t in range(V,0,-1):
        if_less_than_r = np.where(sorted_ratio <= t/V)[0]
        if len(if_less_than_r) <= 4:
            index_by_r[t-1] = 0
        else:
            index_by_r[t-1] = if_less_than_r.max()
    
    index_by_r = index_by_r[index_by_r > 0]    
    index_by_r = np.array([int(i) for i in index_by_r])

    sorted_y_hat = sorted_y_hat[index_by_r]
    sorted_ratio = sorted_ratio[index_by_r]
    
    # rule out points outside of [r_range_low, r_range_up]
    ratio_range_idx_up = sorted_ratio <= r_range_up
    ratio_range_idx_low  = sorted_ratio >= r_range_low
    ratio_range_idx = ratio_range_idx_up * ratio_range_idx_low
    sorted_ratio = sorted_ratio[ratio_range_idx]
    sorted_y_hat = sorted_y_hat[ratio_range_idx]
  
    # solve the least square problem
    ratio_mat = np.zeros((sorted_ratio.shape[0], order+1))
    i=0
    while(i < sorted_ratio.shape[0]):
        r = sorted_ratio[i] 
        for j in range(order +1):
            ratio_mat[i,j] = r**j 
        i+=1
        
    
    id_matrix = np.eye( ratio_mat.shape[1] )

    ratio_mat_T = np.ascontiguousarray(ratio_mat.T)
    ratio_mat = np.ascontiguousarray(ratio_mat)
    RTR = np.ascontiguousarray(ratio_mat_T @ ratio_mat+ id_matrix * lamda)
    RTR_inv = np.ascontiguousarray(np.linalg.inv(RTR))
    sorted_y_hat = np.ascontiguousarray(sorted_y_hat)
    

    return (RTR_inv @ ratio_mat_T @ sorted_y_hat ), all_ratio, all_y_hat, sorted_ratio, sorted_y_hat.ravel() 
   
    
   

class NaiveRegressionEstimator(object):
    """ Naive Regression Estimator
    
    The standard estimator which averages labels in the cell.
    
    
    Parameters
    ----------
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
        
    num_samples : int
        Number of samples in the cell.
        
    dt_X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    V : int
        Parameter for homothetic estimation. The estimations are taken at 
        i/V, i = 1, ..., V.
    
    order : int > 0
        Extrapolation order.
    
    r_range_low : float in [0, 1]
        Lower bound of homothetic ratio to consider.
    
    r_range_up : float in [0, 1], > r_range_low
        Upper bound of homothetic ratio to consider.
        
    
    lamda : float, >0
        Ridge regularization parameter. 
        
    Attributes
    ----------
    y_hat : float
        The final estimation, i.e. mean of dt_Y.
    """
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X, 
                 dt_Y, 
                 order = None,
                 V = None,
                 r_range_up = None,
                 r_range_low = None,
                lamda = None):
        
        self.dt_Y = dt_Y
        self.n_node_samples = dt_X.shape[0]
        self.X_range = X_range
        
        
    def fit(self):
        if self.n_node_samples != 0:
            self.y_hat = self.dt_Y.mean()
        else:
            self.y_hat = 0
        
    def predict(self, test_X):
        y_predict = np.full(test_X.shape[0], self.y_hat)
        return y_predict
    

class ExtraRegressionEstimator(object):
    """ Extrapolation Regression Estimator
  
    
    Parameters
    ----------
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
        
    num_samples : int
        Number of samples in the cell.
        
    dt_X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    V : int
        Parameter for homothetic estimation. The estimations are taken at 
        i/V, i = 1, ..., V.
    
    order : int > 0
        Extrapolation order.
    
    r_range_low : float in [0, 1]
        Lower bound of homothetic ratio to consider.
    
    r_range_up : float in [0, 1], > r_range_low
        Upper bound of homothetic ratio to consider.
        
    
    lamda : float, >0
        Ridge regularization parameter. 
        
   
    """
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X,
                 dt_Y,
                 order,
                 V = 1,
                 r_range_up=1,
                 r_range_low=0,
                lamda=0.01,
                ):
        
        self.X_range = X_range
        self.dim = X_range.shape[1]
        self.dt_X = dt_X
        self.dt_Y = dt_Y
        self.order = order
        self.lamda = lamda
        self.n_node_samples = dt_X.shape[0]
        self.r_range_up = r_range_up
        self.r_range_low = r_range_low
        self.V = V
    
    def fit(self):
        self.y_hat = None
        
    def predict(self, test_X):
        
        """Prediction function

        Parameters
        ----------
        test_X : array-like of shape (n_test_, dim_)
            An array of test points in the cell.
        
        Returns
        -------
        y_predict : array-like of shape (n_test_, )
            The prediction
        
        """
        
        assert self.V!=0
        
        if len(test_X)==0:
            return np.array([])

        pre_vec=[]

        # for each test x, compute extrapolated estimation
        for X in test_X:
            pred_weights, _, _, _, _ = extrapolate_regression(self.dt_X,
                                                                 self.dt_Y, 
                                                                 X, 
                                                                 self.X_range, 
                                                                 self.order,
                                                                 self.r_range_low, 
                                                                 self.r_range_up,
                                                                 self.V, 
                                                                 self.lamda)
            pre_vec.append(pred_weights[0,0])
            
        y_predict=np.array(pre_vec)
        return y_predict
    
    def get_info(self, x):
        """Query the extrapolation information

        Parameters
        ----------
        x : array-like of shape (dim_, 1)
            The extrapolation point.
        
        Returns
        -------
        weights : array-like of shape (order, 1)
            Learnt coefficients of polynomials, weights[0,0] is the final extrapolated 
            estimation. 
            
        all_ratio : array-like of shape (n_sample_, )
            The sorted vector of homothetic ratios associated to all points in 
            the cell. 
            
        all_y_hat : array-like of shape (n_sample_, )
            The sorted vector of homothetic estimations associated to all points in 
            the cell. 
        
        all_ratio : array-like of shape (V, )
            The sorted vector of selected homothetic ratios. 
            
        all_y_hat : array-like of shape (V, )
            The sorted vector of selected homothetic estimations. 
        
        """
        
        assert self.V != 0
        assert len(x.shape) == 2
        x = x.ravel()
        

        pred_weights, all_r , all_y_hat , used_r, used_y_hat = extrapolate_regression(self.dt_X,
                                                                                         self.dt_Y,  
                                                                                         x, 
                                                                                         self.X_range, 
                                                                                         self.order,
                                                                                         self.r_range_low,
                                                                                         self.r_range_up,
                                                                                         self.V,
                                                                                         self.lamda)
        return pred_weights, all_r, all_y_hat, used_r, used_y_hat