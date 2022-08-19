import numpy as np
from scipy.signal import kaiserord, lfilter, firwin


def kaiser_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = None,
    highcut: float = None,
    ripple_db: float = 60.0,
    tranzition_width: float = 5.0,
) -> np.ndarray:
    """
    Determine the filter window parameters for the Kaiser window method.
    Filter data along one-dimension with Kaiser window filter
    Args:
        signal: An N-dimensional input array.
        fs: The sampling frequency of the digital system
        lowcut: lowpass frequency
        highcut: highpass frequency
        ripple_db: Upper bound for the deviation (in dB) of the magnitude of the filter’s frequency
        tranzition_width: Width of transition region
    Returns:
        filtered_signal: The output of the digital filter.
    """
    # The Nyquist rate of the signal.
    nyq_rate = fs * 0.5

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = tranzition_width / nyq_rate

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    if lowcut is None:
        taps = firwin(N, highcut / nyq_rate, window=("kaiser", beta), pass_zero="highpass")
    elif highcut is None:
        taps = firwin(N, lowcut / nyq_rate, window=("kaiser", beta), pass_zero="lowpass")
    else:
        taps = firwin(
            N,
            [lowcut / nyq_rate, highcut / nyq_rate],
            window=("kaiser", beta),
            pass_zero="bandpass",
        )

    # Use lfilter to filter x with the FIR filter.
    filtered_signal = lfilter(taps, 1.0, signal)

    return filtered_signal
