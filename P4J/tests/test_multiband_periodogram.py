import numpy as np
import pandas as pd
import os
from numpy.testing import assert_allclose
import unittest
from P4J import MultiBandPeriodogram


class TestMultibandPeriodogram(unittest.TestCase):
    def setUp(self) -> None:
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(
            dirname,
            'multiband_light_curve.csv'
        )
        df = pd.read_csv(filename)
        self.mjds = df['mjds'].values
        self.mags = df['mags'].values
        self.errs = df['errs'].values
        self.fids = df['fids'].values

    def test_mbperiodogram(self):
        my_per = MultiBandPeriodogram(method="MHAOV")
        my_per.set_data(self.mjds, self.mags, self.errs, self.fids)
        my_per.frequency_grid_evaluation(fmin=0.01, fmax=10., fresolution=1e-4)
        my_per.finetune_best_frequencies(n_local_optima=3, fresolution=1e-5)
        best_freq, best_per = my_per.get_best_frequencies()
        self.assertEqual(len(my_per.per_single_band), len(np.unique(self.fids)))
        assert_allclose(
            best_freq,
            np.array(
                [1.234182, 9.704867, 3.488425],
                dtype=np.float32),
            rtol=1e-4)
        assert_allclose(
            best_per,
            np.array(
                [142.32545, 83.98332, 77.325165],
                dtype=np.float32),
            rtol=1e-2)

    def test_mbperiodogram_optimal_grid(self):
        my_per = MultiBandPeriodogram(method="MHAOV")
        my_per.set_data(self.mjds, self.mags, self.errs, self.fids)
        my_per.optimal_frequency_grid_evaluation(
            smallest_period=0.1,
            largest_period=100.0,
            shift=0.1
        )
        my_per.optimal_finetune_best_frequencies(
            times_finer=10.0, n_local_optima=3)
        best_freq, best_per = my_per.get_best_frequencies()
        self.assertEqual(len(my_per.per_single_band), len(np.unique(self.fids)))
        assert_allclose(
            best_freq,
            np.array(
                [1.234182, 9.704867, 3.488425],
                dtype=np.float32),
            rtol=1e-4)
        assert_allclose(
            best_per,
            np.array(
                [142.32545, 83.98332, 77.325165],
                dtype=np.float32),
            rtol=1e-2)

    def test_mbperiodogram_log_period_grid(self):
        my_per = MultiBandPeriodogram(method="MHAOV")
        my_per.set_data(self.mjds, self.mags, self.errs, self.fids)
        my_per.frequency_grid_evaluation(
            fmin=0.01, fmax=10., fresolution=1e-4, log_period_spacing=True)
        my_per.finetune_best_frequencies(n_local_optima=3, fresolution=1e-5)
        best_freq, best_per = my_per.get_best_frequencies()
        self.assertEqual(len(my_per.per_single_band), len(np.unique(self.fids)))
        assert_allclose(
            best_freq,
            np.array(
                [1.234194, 9.704853, 3.488431],
                dtype=np.float32),
            rtol=1e-4)
        assert_allclose(
            best_per,
            np.array(
                [142.324604, 83.980628, 77.327162],
                dtype=np.float32),
            rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
