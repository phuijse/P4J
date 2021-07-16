import numpy as np
from numpy.testing import assert_allclose
import unittest
from P4J import periodogram
from P4J.generator import synthetic_light_curve_generator


class TestPeriodogram(unittest.TestCase):
    def setUp(self) -> None:
        lc_generator = synthetic_light_curve_generator(T=100.0, N=50, rseed=0)
        lc_generator.set_model(f0=1.23456, A=[1.0, 0.5, 0.25])
        mjd, mag, err = lc_generator.draw_noisy_time_series(SNR=2.0)
        self.mjd = mjd
        self.mag = mag
        self.err = err

    def test_standardize(self):
        my_per = periodogram(method='AOV')
        my_per.set_data(self.mjd, self.mag, self.err, standardize=True)
        assert_allclose(
            my_per.mag[:3],
            np.array([-0.858135,  0.484007,  0.997515], dtype=np.float32),
            atol=1e-4)
        assert_allclose(
            my_per.err[:3],
            np.array([1.422936, 0.715789, 1.490285], dtype=np.float32),
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

    def test_periodogram_mhaov(self):
        best_freq, best_per = self.fit_eval_periodogram('MHAOV')

        self.assertEqual(best_freq.dtype, np.float32)
        self.assertEqual(best_per.dtype, np.float32)
        self.assertEqual(best_freq.shape, (3,))
        self.assertEqual(best_per.shape, (3,))

        assert_allclose(
            best_freq,
            np.array([1.234198, 9.704867, 0.869914], dtype=np.float32),
            rtol=1e-4)
        assert_allclose(
            best_per,
            np.array([61.65133, 43.194977, 38.42828], dtype=np.float32),
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
            np.array([9.660553, 8.129339, 8.114369], dtype=np.float32),
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
            np.array([1.235051, 6.06309, 3.489141], dtype=np.float32),
            rtol=1e-4)
        assert_allclose(
            best_per,
            np.array([0.039919, 0.030958, 0.029757], dtype=np.float32),
            rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
