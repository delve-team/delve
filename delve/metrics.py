import numpy as np


def get_explained_variance(eig_vals, threshold=0.99, return_cum=False):
    """Return number of `eig_vals` needed to explain `threshold`*100 percent of variance
    Args:
        eig_vals   : numpy.ndarray
        threshold  : percent of variance to explain
        return_cum : return the cumulative array of eigenvalues
    Return:
        nr_eig_vals : int
        cum_var_exp : numpy.ndarray

            or

        nr_eig_vals : int
    """
    tot = sum(eig_vals)
    var_exp = [(i / tot) for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    nr_eig_vals = np.argmax(cum_var_exp > threshold)
    if return_cum:
        return nr_eig_vals, cum_var_exp
    return nr_eig_vals

def get_layer_saturation(nr_eig_vals, layer_width):
    saturation = round(100 * nr_eig_vals / layer_width, 2)
    return saturation

def latent_pca(latent_history, subsample_rate=50):
    """Get NxN matrix of sorted (largest to smallest) principal components from `latent_history`
    Args:
        latent_history : list, z-dim bottleneck outputs during autoencoder training
        subsample_rate : int, subsample rate before calculating PCs
        eig_val_only   : bool, return list of eigenvalues
    Returns:
        P              : numpy.ndarray, NxN square matrix of eigenvectors calculated over training
        eig_vals       : numpy.ndarray of eigenvalues

            or

        eig_vals       : numpy.ndarray of eigenvalues, sorted by absolute value in descending order
    """
    history_array = np.vstack(latent_history) # list to array
    if len(history_array.shape) == 4: # conv
        history_array = np.mean(history_array, axis=(2,3))
    history_array = history_array.reshape(history_array.shape[0], -1)
    assert (len(history_array.shape) is 2), "Stacked layer history shape is {}, \
        should be 2".format(history_array.shape)

    embeddings = np.vstack(
        history_array)[::subsample_rate]  # subsample every Nth representation
    cov = np.cov(embeddings.T)

    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                 for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda x: x[0], reverse=True)
    eig_vals, eig_vecs = zip(*eig_pairs)

    P = np.vstack(eig_vecs)
    return np.array(eig_vals), P


def batch_mean(batch):
    """Get mean of first vector in `batch`."""  #TODO: Add support for non-dense layers.
    return np.mean(batch[0])


def batch_cov(batch):
    """Get covariance of first instance in `batch`."""  #TODO: Add support for non-dense layers.
    return np.cov(batch[0])
