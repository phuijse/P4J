import numpy as np
import unittest
from P4J import MultiBandPeriodogram
from P4J.generator import synthetic_light_curve_generator


class TestMultibandPeriodogram(unittest.TestCase):
    def setUp(self) -> None:
        lc_generator = synthetic_light_curve_generator(T=100.0, N=50, rseed=0)
        lc_generator.set_model(f0=1.23456, A=[1.0, 0.5, 0.25])
        mjd1, mag1, err1 = lc_generator.draw_noisy_time_series(SNR=2.0)
        mjd2, mag2, err2 = lc_generator.draw_noisy_time_series(SNR=3.0)
        mjds = np.hstack((mjd1, mjd2))
        mags = np.hstack((mag1, mag2))
        errs = np.hstack((err1, err2))
        fids = np.array([['r']*50 + ['g']*50])[0, :]
        self.mjds = mjds
        self.mags = mags
        self.errs = errs
        self.fids = fids

    def test_mbperiodogram(self):
        my_per = MultiBandPeriodogram(method="MHAOV")
        my_per.set_data(self.mjds, self.mags, self.errs, self.fids)
        my_per.frequency_grid_evaluation(fmin=0.01, fmax=10., fresolution=1e-4)
        my_per.finetune_best_frequencies(n_local_optima=3, fresolution=1e-5)
        best_freq, best_per = my_per.get_best_frequencies()
        self.assertEqual(len(my_per.per_single_band), len(np.unique(self.fids)))
        self.assertTrue(
            np.allclose(
                best_freq,
                np.array([1.2341979, 9.704924, 0.86988854], dtype=np.float32)))
        self.assertTrue(
            np.allclose(
                best_per,
                np.array([125.56125,  81.32925,  74.50633], dtype=np.float32)))


if __name__ == '__main__':
    unittest.main()
