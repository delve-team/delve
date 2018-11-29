import numpy as np


def get_explained_variance(eig_vals, threshold=0.99, return_cum=False):
    """Return number of `eig_vals` needed to explain `threshold`*100 percent of variance
    Args:
        eig_vals   : numpy.ndarray of eigenvalues in descending order
        threshold  : percent of variance to explain
        return_cum : return the cumulative array of eigenvalues
    Return:
        nr_eig_vals : int
        cum_var_exp : numpy.ndarray

            or

        nr_eig_vals : int
    """
    tot = sum(eig_vals)
    var_exp = [(i / tot) for i in eig_vals]
    cum_var_exp = np.cumsum(var_exp)
    nr_eig_vals = np.argmax(cum_var_exp > threshold) + 1  # argmax starts at 0 for 1 eigval
    if return_cum:
        return nr_eig_vals, cum_var_exp
    return nr_eig_vals


def get_layer_saturation(nr_eig_vals, layer_width):
    saturation = round(100 * nr_eig_vals / layer_width, 2)
    return saturation


def _get_cov(layer_history, subsample_rate=10, window_size=100):
    """Get covariance matrix of layer activation history.
    Args:
        subsample_rate : int, subsample rate before calculating PCs
        window_size    : int, how many activations to use for calculating principal components
    """
    history_array = np.vstack(layer_history[-1])  # use only current activation
    idx = np.random.randint(0, history_array.shape[0], history_array.shape[0]//subsample_rate + 1)
    embeddings = history_array[idx]  # select few samples for batch to speed up the calculations
    embeddings = np.reshape(embeddings, (embeddings.shape[1], -1))

    cov = np.cov(embeddings)
    return cov


def latent_pca(layer_history):
    """Get NxN matrix of principal components sorted in descending order from `layer_history`
    Args:
        layer_history : list, layer outputs during training
    Returns:
        eig_vals       : numpy.ndarray of absolute value of eigenvalues, sorted in descending order
        P              : numpy.ndarray, NxN square matrix of principal components calculated over training

    """
    cov = _get_cov(layer_history)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda x: x[0], reverse=True)
    eig_vals, eig_vecs = zip(*eig_pairs)

    P = np.vstack(eig_vecs)
    return np.array(eig_vals), P


def latent_svd(layer_history):
    """Get NxN matrix of sorted (largest to smallest) singular values from `layer_history`
    Args:
        layer_history : list, layer outputs during training
    Returns:
        eig_vals  : numpy.ndarray of eigenvalues, sorted in descending order
        eig_vecs   : numpy.ndarray, NxN square matrix of eigenvectors calculated over training

    NOTE: Method is experimental.
    """
    cov = _get_cov(layer_history)
    u, s, eig_vecs = np.linalg.svd(cov)
    eig_vals = s ** 2
    return eig_vals, eig_vecs


def batch_mean(batch):
    """Get mean of first vector in `batch`."""  # TODO: Add support for non-dense layers.
    return np.mean(batch[0])


def batch_cov(batch):
    """Get covariance of first instance in `batch`."""  # TODO: Add support for non-dense layers.
    return np.cov(batch[0])
