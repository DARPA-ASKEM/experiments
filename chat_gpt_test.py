import numpy as np
from scipy.signal import find_peaks

def find_periodic_signals_autocorr(data, max_lag=None, threshold=None):
    """
    Find periodic signals in a dataset using autocorrelation.

    Parameters
    ----------
    data : ndarray
        The input data to search for periodic signals.
    max_lag : int, optional
        The maximum lag to compute the autocorrelation for.
        If not specified, the default value is len(data) // 2.
        If the length of data is less than 100, max_lag is set to len(data) - 1.
    threshold : float, optional
        The minimum height of a peak in the autocorrelation function
        to be considered a periodic signal.
        If not specified, the default value is 0.1 * np.max(np.abs(autocorr)).

    Returns
    -------
    freqs : ndarray
        The frequencies of the periodic signals found in the data.
    amps : ndarray
        The amplitudes of the periodic signals found in the data.
    """
    if max_lag is None:
        if len(data) < 100:
            max_lag = len(data) - 1
        else:
            max_lag = len(data) // 2

    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    freqs = np.arange(1, max_lag + 1)
    amps = np.abs(autocorr[1:max_lag+1])
    
    # Normalize the amplitudes
    amps /= np.max(amps)

    if threshold is None:
        threshold = 0.1 * np.max(np.abs(autocorr))

    # Find peaks in the autocorrelation function
    peaks, _ = find_peaks(amps, height=threshold)

    return freqs[peaks], amps[peaks]