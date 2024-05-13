import numpy as np
import warnings

def get_eigenvector_product(gamma, eigvals, n_components, n_features):
    """
    analytical solution to eigenvector product according to Paul, 2007
    """
    n_data_train = int(n_features / gamma)
    n_spikes = len(eigvals)
    # critical point / threshold
    threshold = 1 + np.sqrt(gamma)
    # compute analytic solution
    diag = (1 - gamma / (eigvals - 1) ** 2) / (1 + gamma / (eigvals - 1))
    diag[eigvals < threshold] = 0

    # remove values if n_components is chosen too small
    if n_components < n_spikes:
        diag[n_components:] = 0

    # add stuff if n_components is chosen too large
    l = min(n_components, n_features, n_data_train)
    if l > n_spikes:
        # assume that all entries are equally distributed on average
        add_part = (1 - diag) / (n_features - n_spikes)
        diag += (l - n_spikes) * add_part
    return np.diag(diag)


def get_eigenvalue_dist(gamma, eigvals, n_data=None):
    """
    analytical solution to eigenvalue distribution according to Paul, 2007
    """
    threshold = 1 + np.sqrt(gamma)

    # for eigvals > threshold -> Normal distributed
    mu = gamma * eigvals / (eigvals - 1) + eigvals

    # for eigvals <= threshold -> tracy widom distributed with lambda = mu + n**-2/3 * sigma * TW1
    mu[eigvals <= threshold] = threshold ** 2

    return mu

def get_marchenko_pastur(x_mp, gamma, eps=0., normalize=False):
    # support
    gamma_n, gamma_p = (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2
    # compute MP distribution
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if gamma <= 1:
            mp = 1 / (2 * np.pi) * np.sqrt((gamma_p - x_mp) * (x_mp - gamma_n)) / (gamma * x_mp)
        else:
            mp = 1 / (2 * np.pi) * np.sqrt((gamma_p - x_mp) * (x_mp - gamma_n)) / (gamma * x_mp)
            mp[x_mp == eps] = (1 - 1 / gamma) * 2 / (x_mp[1] - x_mp[0])
            # last part is scaling factor for correct normalization
    mp[np.isnan(mp)] = 0
    if normalize:
        # normalize distribution
        mp /= np.trapz(mp, x_mp)
    return mp


def get_cumulative_marchenko_pastur(x_mp, gamma):
    # support
    gamma_n, gamma_p = (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2

    def r_(x):
        return np.sqrt((gamma_p - x) / (x - gamma_n))

    def F_(x):
        temp1 = np.pi * gamma
        temp2 = np.sqrt((gamma_p - x) * (x - gamma_n))
        temp3 = (1 + gamma) * np.arctan((r_(x) ** 2 - 1) / (2 * r_(x)))
        temp4 = (1 - gamma) * np.arctan((gamma_n * r_(x) ** 2 - gamma_p) / (2 * (1 - gamma) * r_(x)))
        return 1 / (2 * np.pi * gamma) * (temp1 + temp2 - temp3 + temp4)

    # compute cummulative MP distribution
    mp = np.zeros_like(x_mp)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if gamma <= 1:
            # mp takes value 1 if >= gamma_n and 1 if < gamma_n
            mp[(x_mp >= gamma_n) * (x_mp < gamma_p)] = F_(x_mp[(x_mp >= gamma_n) * (x_mp < gamma_p)])
        else:
            # dirac part
            mp[x_mp < gamma_n] = (gamma - 1) / gamma
            # bulk part
            mp[(x_mp >= gamma_n) * (x_mp < gamma_p)] = F_(x_mp[(x_mp >= gamma_n) * (x_mp < gamma_p)]) \
                                                       + (gamma - 1) / (2 * gamma)
    # all values above gamma_p are 1
    mp[x_mp >= gamma_p] = 1
    return mp