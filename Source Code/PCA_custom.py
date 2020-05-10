def PCA_custom(data, dims_rescaled_data=100):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as np
    from scipy import linalg as la
    # m, n = data.shape
    # mean center the datam
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    r = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since r is symmetric,
    # the performance gain is substantial
    evals, evecs = la.eigh(r)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return evecs, np.dot(evecs.T, data.T).T, evals
