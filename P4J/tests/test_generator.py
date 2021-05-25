import numpy as np
import unittest
from P4J.generator import synthetic_light_curve_generator


class TestGenerator(unittest.TestCase):
    def test_generator_draw(self):
        lc_generator = synthetic_light_curve_generator(T=100.0, N=100, rseed=0)
        lc_generator.set_model(f0=1.23456, A=[1.0])
        mjd, mag, err = lc_generator.draw_noisy_time_series(SNR=20.0)
        self.assertEqual(mjd.shape, (100,))
        self.assertEqual(mag.shape, (100,))
        self.assertEqual(err.shape, (100,))
        self.assertTrue(np.allclose(lc_generator.get_fundamental_frequency(), 1.23456))


if __name__ == '__main__':
    unittest.main()
