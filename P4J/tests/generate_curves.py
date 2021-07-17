import numpy as np
import pandas as pd

from P4J.generator import synthetic_light_curve_generator


def create_single_band_light_curve(filename):
    lc_generator = synthetic_light_curve_generator(T=100.0, N=50, rseed=0)
    lc_generator.set_model(f0=1.23456, A=[1.0, 0.5, 0.25])
    mjd, mag, err = lc_generator.draw_noisy_time_series(SNR=2.0)
    df = pd.DataFrame()
    df['mjd'] = mjd
    df['mag'] = mag
    df['err'] = err
    df.to_csv(filename, index=False)


def create_multiband_light_curve(filename):
    lc_generator = synthetic_light_curve_generator(T=100.0, N=50, rseed=0)
    lc_generator.set_model(f0=1.23456, A=[1.0, 0.5, 0.25])
    mjd1, mag1, err1 = lc_generator.draw_noisy_time_series(SNR=2.0)
    mjd2, mag2, err2 = lc_generator.draw_noisy_time_series(SNR=3.0)
    mjds = np.hstack((mjd1, mjd2))
    mags = np.hstack((mag1, mag2))
    errs = np.hstack((err1, err2))
    fids = np.array([['r'] * 50 + ['g'] * 50])[0, :]

    df = pd.DataFrame()
    df['mjds'] = mjds
    df['mags'] = mags
    df['errs'] = errs
    df['fids'] = fids
    df.to_csv(filename, index=False)


'''Believe it or not, the output of this code depends on the platform despite 
setting a seed for the random number generator.'''

create_single_band_light_curve('single_band_light_curve.csv')
create_multiband_light_curve('multiband_light_curve.csv')
