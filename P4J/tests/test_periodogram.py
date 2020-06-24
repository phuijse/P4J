import numpy as np
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
        self.assertTrue(
            np.allclose(
                my_per.mag[:3],
                np.array([-0.885704, 0.47217235, 0.9916998], dtype=np.float32),
                atol=1e-7))
        self.assertTrue(
            np.allclose(
                my_per.err[:3],
                np.array([1.439618, 0.72418076, 1.507756], dtype=np.float32),
                atol=1e-7))

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

        self.assertTrue(
            np.allclose(
                best_freq,
                np.array([1.234198, 9.704867, 3.4884243], dtype=np.float32), rtol=1e-5))
        self.assertTrue(
            np.allclose(
                best_per,
                np.array([66.69726, 42.329742, 38.671776], dtype=np.float32), rtol=1e-5))

    def test_periodogram_aov(self):
        best_freq, best_per = self.fit_eval_periodogram('AOV')

        self.assertEqual(best_freq.dtype, np.float32)
        self.assertEqual(best_per.dtype, np.float32)
        self.assertEqual(best_freq.shape, (3,))
        self.assertEqual(best_per.shape, (3,))

        self.assertTrue(
            np.allclose(
                best_freq,
                np.array([1.2353979, 4.0710936, 9.823684], dtype=np.float32),
                rtol=1e-5))
        self.assertTrue(
            np.allclose(
                best_per,
                np.array([9.226295, 8.598029, 7.567729], dtype=np.float32),
                rtol=1e-5))

    def test_periodogram_pdm(self):
        best_freq, best_per = self.fit_eval_periodogram('PDM1')

        self.assertEqual(best_freq.dtype, np.float32)
        self.assertEqual(best_per.dtype, np.float32)
        self.assertEqual(best_freq.shape, (3,))
        self.assertEqual(best_per.shape, (3,))

        self.assertTrue(
            np.allclose(
                best_freq,
                np.array([1.2352979, 1.235708, 1.234098], dtype=np.float32),
                rtol=1e-5))
        self.assertTrue(
            np.allclose(
                best_per,
                np.array([-0.20746106, -0.23871078, -0.27404344], dtype=np.float32),
                rtol=1e-5))

    def test_periodogram_qmi(self):
        best_freq, best_per = self.fit_eval_periodogram('QMIEU')

        self.assertEqual(best_freq.dtype, np.float32)
        self.assertEqual(best_per.dtype, np.float32)
        self.assertEqual(best_freq.shape, (3,))
        self.assertEqual(best_per.shape, (3,))

        self.assertTrue(
            np.allclose(
                best_freq,
                np.array([1.234968, 3.4893143, 6.06346], dtype=np.float32),
                rtol=1e-5))
        self.assertTrue(
            np.allclose(
                best_per,
                np.array([0.03973941, 0.03063367, 0.02963623], dtype=np.float32),
                rtol=1e-5))


if __name__ == '__main__':
    unittest.main()
