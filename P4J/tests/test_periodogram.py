import numpy as np
from numpy.testing import assert_allclose
from P4J import periodogram
from P4J.generator import synthetic_light_curve_generator

def create_test_data():
    lc_generator = synthetic_light_curve_generator(T=100.0, N=50, rseed=0)
    lc_generator.set_model(f0=1.23456, A=[1.0, 0.5, 0.25])
    mjd, mag, err = lc_generator.draw_noisy_time_series(SNR=2.0)
    return mjd, mag, err

def test_standardize(tol=1e-7):
    my_per = periodogram(method='AOV')
    mjd, mag, err = create_test_data()
    my_per.set_data(mjd, mag, err, standardize=True)
    assert_allclose(my_per.mag[:3], np.array([-0.87692106, 0.44308114, 0.94811785], dtype=np.float32), atol=tol)
    assert_allclose(my_per.err[:3], np.array([1.399464   , 0.70398176, 1.4657015], dtype=np.float32), atol=tol)
    
def test_removenan():
    my_per = periodogram(method='AOV')
    mjd, mag, err = create_test_data()
    mjd[1], mag[5], err[8] = np.nan, np.nan, np.nan
    my_per.set_data(mjd, mag, err, remove_nan=True)
    assert len(my_per.mag) == len(mjd)-3    

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
    assert_allclose(best_freq, np.array([1.234198 , 9.704909, 6.981068], dtype=np.float32))
    assert_allclose(best_per, np.array([58.99221, 40.944813, 37.43901], dtype=np.float32))
    
def test_periodogram_aov():
    best_freq, best_per = fit_eval_periodogram('AOV')

    assert best_freq.dtype == np.float32
    assert best_per.dtype == np.float32
    assert best_freq.shape == (3,)
    assert best_per.shape == (3,)
    assert_allclose(best_freq, np.array([1.2353979, 4.0710936, 9.823684], dtype=np.float32))
    assert_allclose(best_per, np.array([9.8947315, 8.2189865, 8.14342], dtype=np.float32))
    
def test_periodogram_pdm():
    best_freq, best_per = fit_eval_periodogram('PDM1')

    assert best_freq.dtype == np.float32
    assert best_per.dtype == np.float32
    assert best_freq.shape == (3,)
    assert best_per.shape == (3,)
    assert_allclose(best_freq, np.array([1.2352979, 1.235708, 1.234098], dtype=np.float32))
    assert_allclose(best_per, np.array([-0.19907248, -0.23332588, -0.28197145], dtype=np.float32))

def test_periodogram_qmi():
    best_freq, best_per = fit_eval_periodogram('QMIEU')
    
    assert best_freq.dtype == np.float32
    assert best_per.dtype == np.float32
    assert best_freq.shape == (3,)
    assert best_per.shape == (3,)
    assert_allclose(best_freq, np.array([1.234878, 6.06321, 3.4890144], dtype=np.float32))
    assert_allclose(best_per, np.array([0.03774722, 0.029475132, 0.028804852], dtype=np.float32))



