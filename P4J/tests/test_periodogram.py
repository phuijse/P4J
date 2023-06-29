import numpy as np
import os

import pandas as pd
from numpy.testing import assert_allclose
import unittest
from P4J import periodogram


class TestPeriodogram(unittest.TestCase):
    def setUp(self) -> None:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(
            dirname,
            'single_band_light_curve.csv'
        )
        df = pd.read_csv(filename)
        self.mjd = df['mjd'].values
        self.mag = df['mag'].values
        self.err = df['err'].values

    def test_standardize(self):
        my_per = periodogram(method='AOV')
        my_per.set_data(self.mjd, self.mag, self.err, standardize=True)
        assert_allclose(
            my_per.mag[:3],
            np.array([-0.885704,  0.472172,  0.9917], dtype=np.float32),
            atol=1e-4)
        assert_allclose(
            my_per.err[:3],
            np.array([1.439618, 0.724181, 1.507756], dtype=np.float32),
            atol=1e-4)

    def test_removenan(self):
        my_per = periodogram(method='AOV')
        self.mjd[1], self.mag[5], self.err[8] = np.nan, np.nan, np.nan
        my_per.set_data(self.mjd, self.mag, self.err, remove_nan=True)
        self.assertEqual(len(my_per.mag), len(self.mjd) - 3)

    def fit_eval_periodogram(self, method):
        my_per = periodogram(method=method)
        my_per.set_data(self.mjd, self.mag, self.err)
        my_per.frequency_grid_evaluation(fmin=0.01, fmax=10., fresolution=1e-4)
        my_per.finetune_best_frequencies(n_local_optima=3, fresolution=1e-5)
        return my_per.get_best_frequencies()

    def test_optimal_frequency_grid_evaluation(self):
        my_per = periodogram(method='MHAOV')
        my_per.set_data(self.mjd, self.mag, self.err)
        my_per.optimal_frequency_grid_evaluation(
            smallest_period=0.1,
            largest_period=100.0,
            shift=0.1
        )
        my_per.optimal_finetune_best_frequencies(10.0, n_local_optima=3)
        best_freq, best_per = my_per.get_best_frequencies()

        self.assertEqual(best_freq.dtype, np.float32)
        self.assertEqual(best_per.dtype, np.float32)
        self.assertEqual(best_freq.shape, (3,))
        self.assertEqual(best_per.shape, (3,))

        assert_allclose(
            best_freq[0],
            1.23456,
            rtol=1e-3)

    def test_periodogram_mhaov(self):
        best_freq, best_per = self.fit_eval_periodogram('MHAOV')

        self.assertEqual(best_freq.dtype, np.float32)
        self.assertEqual(best_per.dtype, np.float32)
        self.assertEqual(best_freq.shape, (3,))
        self.assertEqual(best_per.shape, (3,))

        assert_allclose(
            best_freq,
            np.array([1.234198, 9.704867, 3.488431], dtype=np.float32),
            rtol=1e-4)
        assert_allclose(
            best_per,
            np.array([66.697205, 42.329742, 38.673233], dtype=np.float32),
            rtol=1e-4)

    def test_periodogram_aov(self):
        best_freq, best_per = self.fit_eval_periodogram('AOV')

        self.assertEqual(best_freq.dtype, np.float32)
        self.assertEqual(best_per.dtype, np.float32)
        self.assertEqual(best_freq.shape, (3,))
        self.assertEqual(best_per.shape, (3,))

        assert_allclose(
            best_freq,
            np.array([1.2353979, 4.0710936, 9.823684], dtype=np.float32),
            rtol=1e-4)

        assert_allclose(
            best_per,
            np.array([9.226296, 8.59803, 7.567729], dtype=np.float32),
            rtol=1e-4)

    def test_periodogram_pdm(self):
        best_freq, best_per = self.fit_eval_periodogram('PDM1')

        self.assertEqual(best_freq.dtype, np.float32)
        self.assertEqual(best_per.dtype, np.float32)
        self.assertEqual(best_freq.shape, (3,))
        self.assertEqual(best_per.shape, (3,))

        np.allclose(
            best_freq,
            np.array([1.2352979, 1.235708, 1.234098], dtype=np.float32),
            rtol=1e-5)
        np.allclose(
            best_per,
            np.array([-0.20746106, -0.23871078, -0.27404344], dtype=np.float32),
            rtol=1e-5)

    def test_periodogram_qmi(self):
        best_freq, best_per = self.fit_eval_periodogram('QMIEU')

        self.assertEqual(best_freq.dtype, np.float32)
        self.assertEqual(best_per.dtype, np.float32)
        self.assertEqual(best_freq.shape, (3,))
        self.assertEqual(best_per.shape, (3,))

        assert_allclose(
            best_freq,
            np.array([1.234961, 3.4893, 6.063464], dtype=np.float32),
            rtol=1e-4)
        assert_allclose(
            best_per,
            np.array([0.039739, 0.030634, 0.029636], dtype=np.float32),
            rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
