import numpy as np
from P4J.generator import synthetic_light_curve_generator


def test_generator_draw():
    lc_generator = synthetic_light_curve_generator(T=100.0, N=100, rseed=0)
    lc_generator.set_model(f0=1.23456, A=[1.0])
    mjd, mag, err = lc_generator.draw_noisy_time_series(SNR=20.0)
    assert mjd.shape == (100,)
    assert mag.shape == (100,)
    assert err.shape == (100,)
    assert np.allclose(lc_generator.get_fundamental_frequency(), 1.23456)


if __name__ == '__main__':
    test_generator_draw()
