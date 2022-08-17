import pandas as pd
from otlang.sdk.syntax import Keyword, Positional, OTLType
from pp_exec_env.base_command import BaseCommand, Syntax
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

class FirFilterCommand(BaseCommand):
    # define syntax of your command here
    syntax = Syntax(
        [
            Positional("signal", required=True, otl_type=OTLType.TEXT),
            Keyword("fs", required=True, otl_type=OTLType.NUMERIC),
            Keyword("lowcut", required=True, otl_type=OTLType.NUMERIC),
            Keyword("highcut", required=True, otl_type=OTLType.NUMERIC),
        ],
    )
    use_timewindow = False  # Does not require time window arguments
    idempotent = True  # Does not invalidate cache

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_progress('Start fir_filter command')
        signal_name = self.get_arg("signal").value
        fs = self.get_arg("fs").value
        lowcut = self.get_arg("lowcut").value
        highcut = self.get_arg("highcut").value
        raw_signal = df[signal_name].values

        filtered_signal = kaiser_filter(raw_signal, fs=fs, lowcut=lowcut, highcut=highcut)
        self.log_progress('First part is complete.', stage=1, total_stages=1)

        df[f"filtered_{signal_name}"] = filtered_signal
        return df
