import numpy as np
def g(x):
    return np.tanh(x)
def g_der(x):
    return 1 - g(x) * g(x)

def center(X):
    X = np.array(X)

    mean = X.mean(axis=1, keepdims=True)

    return X - mean

def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten


def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    X = whitening(X)
    components_nr = X.shape[0]
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)
    for i in range(components_nr):
        w = np.random.rand(components_nr)
    for j in range(iterations):
        w_new = calculate_new_w(w, X)
        if i >= 1:
            w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
        distance = np.abs(np.abs((w * w_new).sum()) - 1)
        w = w_new
        if distance < tolerance:
            break
    W[i, :] = w
S = np.dot(W, X)
return S
