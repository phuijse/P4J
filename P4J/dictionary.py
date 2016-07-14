from __future__ import division, print_function
import numpy as np

"""
Generates a multi-harmonic dictionary based on a time vector (t in Nx1) and a fundamental frequency (f0)
"""
def harmonic_dictionary(t, f0, M=1):
    N = len(t)
    Phi = np.ones(shape=(N, 2*M+1))
    for k in range(0, M):
        Phi[:, k+1] = np.cos(2.0*np.pi*t*f0*(k+1))
        Phi[:, M+k+1] = np.sin(2.0*np.pi*t*f0*(k+1))
    return Phi
