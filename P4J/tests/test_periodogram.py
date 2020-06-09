import numpy as np
from P4J import periodogram
from P4J.generator import synthetic_light_curve_generator

def create_test_data():
    lc_generator = synthetic_light_curve_generator(T=100.0, N=50, rseed=0)
    lc_generator.set_model(f0=1.23456, A=[1.0, 0.5, 0.25])
    mjd, mag, err = lc_generator.draw_noisy_time_series(SNR=2.0)
    return mjd, mag, err

def fit_eval_periodogram(method):
    my_per = periodogram(method=method)
    mjd, mag, err = create_test_data()
    my_per.set_data(mjd, mag, err)
    my_per.frequency_grid_evaluation(fmin=0.01, fmax=10., fresolution=1e-4)
    my_per.finetune_best_frequencies(n_local_optima=3, fresolution=1e-5)
    return my_per.get_best_frequencies()

def test_periodogram_mhaov():
    best_freq, best_per = fit_eval_periodogram('MHAOV')

    assert best_freq.dtype == np.float32
    assert best_per.dtype == np.float32
    assert best_freq.shape == (3,)
    assert best_per.shape == (3,)
    assert np.allclose(best_freq, np.array([1.23421 , 9.704909, 6.98109], dtype=np.float32))
    assert np.allclose(best_per, np.array([58.992764, 40.944824, 37.43927], dtype=np.float32))
    
def test_periodogram_pdm():
    best_freq, best_per = fit_eval_periodogram('PDM1')

    assert best_freq.dtype == np.float32
    assert best_per.dtype == np.float32
    assert best_freq.shape == (3,)
    assert best_per.shape == (3,)
    assert np.allclose(best_freq, np.array([1.23531, 1.2345, 1.4456], dtype=np.float32))
    assert np.allclose(best_per, np.array([-0.1723166 , -0.18674569, -0.1979735], dtype=np.float32))

def test_periodogram_pdm():
    best_freq, best_per = fit_eval_periodogram('QMIEU')
    
    assert best_freq.dtype == np.float32
    assert best_per.dtype == np.float32
    assert best_freq.shape == (3,)
    assert best_per.shape == (3,)
    assert np.allclose(best_freq, np.array([1.23488, 6.06321, 3.489], dtype=np.float32))
    assert np.allclose(best_per, np.array([0.03774682, 0.02947513, 0.02880505], dtype=np.float32))



