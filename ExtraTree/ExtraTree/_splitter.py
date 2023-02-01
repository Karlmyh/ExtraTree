import numpy as np
from ._criterion import gini, mse

criterion_func = {"gini":gini,
                  "mse":mse}

class PurelyRandomSplitter(object):    
    """Purely random splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
    
    max_features : float in (0,1]
        Proportion of dimensions to consider when splitting. 
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, random_state = None, max_features = 1.0, search_number = None, threshold = None):
        self.random_state = random_state
        np.random.seed(self.random_state)
        
    def __call__(self, X, X_range, dt_Y = None):
        n_node_samples, dim = X.shape
        
        # randomly choose dimension and split point
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = np.random.uniform(rddim_min, rddim_max)
        return rd_dim, rd_split
    
    
class MidPointRandomSplitter(object):
    """Random mid-point splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
    
    max_features : float in (0,1]
        Proportion of dimensions to consider when splitting. 
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, random_state = None, max_features = 1.0, search_number = None, threshold = None):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_features = max_features
        
    def __call__(self, X, X_range, dt_Y = None):
        n_node_samples, dim = X.shape
        
        # randomly choose a dimension and split at mid-point
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min+ rddim_max)/2
        return rd_dim, rd_split
    
    
class MaxEdgeRandomSplitter(object):
    """Random max-edge splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
    
    max_features : float in (0,1]
        Proportion of dimensions to consider when splitting. 
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, random_state = None, max_features = 1.0, search_number = None, threshold = None):
        self.random_state = random_state
        self.max_features = max_features
        np.random.seed(self.random_state)
        
    def __call__(self, X, X_range ,dt_Y = None):
        n_node_samples, dim = X.shape
        
        # randomly choose among the dimensions with longest edge
        edge_ratio = X_range[1] - X_range[0]
        
        # sub-sample a subset of dimensions
        subsampled_idx = np.random.choice(edge_ratio.shape[0], int(np.ceil(edge_ratio.shape[0] * self.max_features)),replace = False)
        rd_dim = np.random.choice(np.where(edge_ratio[subsampled_idx] == edge_ratio[subsampled_idx].max())[0])
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min + rddim_max)/2
        return rd_dim, rd_split
    
    
    
    
    
    
    
    
    
    
class GainReductionSplitter(object):
    """Abstract information gain based splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
    
    max_features : float in (0,1]
        Proportion of dimensions to consider when splitting. 
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, criterion, random_state = None, max_features = 1.0, search_number = 10, threshold = None):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_features = max_features
        self.search_number = search_number
        self.compute_criterion_reduction = criterion_func[criterion]
        self.threshold = threshold
        
    def __call__(self, X, X_range, dt_Y):
        n_node_samples, dim = X.shape
        
        # sub-sample a subset of dimensions
        subsampled_idx = np.random.choice(dim, int(np.ceil( dim * self.max_features) ), replace = False)

        max_criterion_reduction = np.inf
        split_dim = None
        split_point = None
        
        # search for dimension and split point with maximum criterion reduction
        for d in subsampled_idx:
            
            dt_X_dim_unique = np.unique(X[:,d])
            sorted_split_point = np.unique( np.quantile( dt_X_dim_unique, [(2 * i + 1)/(2 * self.search_number) for i in range(self.search_number) ] ) )
            
            for split in sorted_split_point:
                
                # compute criterion
                criterion_reduction = self.compute_criterion_reduction(X, dt_Y, d, split)
            
                # hault if reduction is small
                if criterion_reduction < max_criterion_reduction and criterion_reduction >= self.threshold:
                    
                    max_criterion_reduction = criterion_reduction
                    split_dim = d
                    split_point = split
            
                

        return split_dim, split_point
    

class MSEReductionSplitter(GainReductionSplitter):
    """MSE reduction splitter class.
    """
    def __init__(self, random_state = None, max_features = 1.0, search_number = 10, threshold = None):
        super(MSEReductionSplitter, self).__init__( criterion = "mse", 
                                                   random_state = random_state, 
                                                   max_features = max_features, 
                                                   search_number = search_number,
                                                   threshold = threshold)
    
    
class GINIReductionSplitter(GainReductionSplitter):
    """GINI reduction splitter class.
    """
    def __init__(self, random_state = None, max_features = 1.0, search_number = 10, threshold = None):
        super(GINIReductionSplitter, self).__init__( criterion = "gini", 
                                                   random_state = random_state, 
                                                   max_features = max_features, 
                                                   search_number = search_number,
                                                   threshold = threshold)
        
        
        










class GainReductionMaxEdgeSplitter(object):
    """Abstract information gain based mid-point splitter class.

    Parameters
    ----------
    random_state : int
        Random state for dimension subsampling and splitting.
    
    max_features : float in (0,1]
        Proportion of dimensions to consider when splitting. 
        
    search_number : int
        Number of points to search on when looking for best split point.
        
    threshold : float in [0, infty]
        Threshold for haulting when criterion reduction is too small.
        
    X : array-like of shape (n_sample_, dim_)
        An array of points in the cell.
    
    dt_Y : array-like of shape (n_sample_, )
        An array of labels in the cell.
        
    X_range : array-like of shape (2, dim_)
        Boundary of the cell, X_range[0, d] and X_range[1, d] stands for the
        lower and upper bound of d-th dimension.
    
    
    Returns
    -------
    rd_dim : int in 0, ..., dim - 1
        The splitting dimension.
        
    rd_split : float
        The splitting point.

    """
    def __init__(self, criterion, random_state = None, max_features = 1.0, search_number = None, threshold = None):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.max_features = max_features
        self.compute_criterion_reduction = criterion_func[criterion]
        self.threshold = threshold
        
    def __call__(self, X, X_range, dt_Y):
        n_node_samples, dim = X.shape
        
        
        edge_ratio = X_range[1] - X_range[0]
        
        # sub-sample a subset of dimensions
        subsampled_idx = np.random.choice(edge_ratio.shape[0], int(np.ceil(edge_ratio.shape[0] * self.max_features)),replace = False)
        max_edges = np.where(edge_ratio[subsampled_idx] == edge_ratio[subsampled_idx].max())[0]
        

        max_criterion_reduction = np.inf
        split_dim = None
        split_point = None
        
        # search for dimension with maximum criterion reduction
        for rd_dim in max_edges:
            
            
            split = ( X_range[1,rd_dim] + X_range[0,rd_dim])/2
                
            # compute criterion
            criterion_reduction = self.compute_criterion_reduction(X, dt_Y, rd_dim, split)
            
            # hault if reduction is small
            if criterion_reduction < max_criterion_reduction and criterion_reduction >= self.threshold:
                
                max_criterion_reduction = criterion_reduction
                split_dim = rd_dim
                split_point = split
            
                

        return split_dim, split_point
    


class MSEReductionMaxEdgeSplitter(GainReductionSplitter):
    """MSE reduction mid-point splitter class.
    """
    def __init__(self, random_state = None, max_features = 1.0, search_number = None, threshold = None):
        super(MSEReductionMaxEdgeSplitter, self).__init__( criterion = "mse", 
                                                   random_state = random_state, 
                                                   max_features = max_features, 
                                                   search_number = search_number,
                                                   threshold = threshold)
    
    
class GINIReductionMaxEdgeSplitter(GainReductionSplitter):
    """GINI reduction mid-point splitter class.
    """
    def __init__(self, random_state = None, max_features = 1.0, search_number = None, threshold = None):
        super(GINIReductionMaxEdgeSplitter, self).__init__( criterion = "gini", 
                                                   random_state = random_state, 
                                                   max_features = max_features, 
                                                   search_number = search_number,
                                                   threshold = threshold)
        
        
        
  