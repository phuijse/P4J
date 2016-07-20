from __future__ import division, print_function
import numpy as np


def irregular_sampling(T, N, rseed=None):
    """
    Generates an irregularly sampled time vector with specified number of 
    samples (N) and time span (T)
    """
    sampling_period = (T/N)
    N = int(N)
    np.random.seed(rseed)    
    t = np.linspace(0, T, num=5*N)
    # First we add jitter
    t[1:-1] += sampling_period*0.5*np.random.randn(5*N-2)
    # Then we do a random permutation and keep only N points 
    P = np.random.permutation(5*N)
    return np.sort(t[P[:N]])
    

def trigonometric_model(t, f0, A):
    """
    Generates a normalized trigonometric model, A corresponds to a vector
    having the amplitudes of the model and t is a time vector, f0 is the 
    fundamental frequency of the model
    """
    y = 0.0
    M = len(A)
    var_y = 0.0
    for k in range(0, M):
        y += A[k]*np.sin(2.0*np.pi*t*f0*(k+1))
        var_y += 0.5*A[k]**2
    return y/np.sqrt(var_y)

def first_order_markov_process(t, variance, time_scale):
    """
    To model the red noise we use a first order markov process, we 
    compute the covariance matrix and sample from a multivariate random
    normal number generator
    """
    N = len(t)
    mu = np.zeros(shape=(N,))
    if variance == 0.0:
        return mu
    dt = np.repeat(np.reshape(t, (1, -1)), N, axis=0)
    dt = np.absolute(dt - dt.T)  # This is NxN
    S = variance*np.exp(-np.absolute(dt)/time_scale)
    return np.random.multivariate_normal(mu, S)

def power_law_noise(t, variance):
    """
    Lets consider a power law with alpha=-2.0, i.e. red noise (pink would
    be -1.0, and white would be 0.0). In this case the PSD is S(f) = A/f**2
    where A is a constant (the PSD is real and even). Then
    rxx(tau) = sum A cos(2pi f tau)/f**2
    If we assume Fs = N/T, df = 1/T (no oversampling), f[k] = k/T then
    rxx(tau) = cT sum cos(2pi k tau/T)/k**2
    The basel problem gives as sum (1/k**2) approx pi**2/6, hence we can
    set c according to the desired variance at lag=0 
    We can compute a correlation matrix and then draw from a multivariate
    normal distribution to obtain a noise vector
    """
    N = len(t)    
    T = (t[-1] - t[0])
    c = (6.0*variance)/(T*np.pi**2)
    f = np.arange(1.0/T, 0.5*N/T, step=1.0/T)  # We ommit f=0.0
    k = f*T
    dt = np.repeat(np.reshape(t, (1, -1)), N, axis=0)
    dt = np.absolute(dt - dt.T)  # This is NxN
    S = np.zeros(shape=(N,N))
    for i in range(0, N):
        for j in range(i, N):
            S[i, j] = np.sum(np.cos(2.0*np.pi*k*dt[i, j]/T)/k**2)*T*c
            S[j, i] = S[i, j]
    mu = np.zeros(shape=(N,))
    return np.random.multivariate_normal(mu, S)
    

def contaminate_time_series(t, y, SNR=10.0, red_noise_ratio=0.5, outlier_ratio=0.0, rseed=None):
    """
    A function that takes an irregulary sampled time series (t, y) 
    and contaminates it, simulating the noise present in light curves. 
    It returns (y, y_noisy, s), where s are the uncertainties 
    associated to the magnitudes (y_noisy). y will be reescaled to fit a 
    given SNR. 
    
    SNR -- Signal to Noise ratio of the resulting time series in dB. SNR }
    is defined as SNR = 10*log(var_signal/var_noise), so
     SNR var_signal/var_noise
     10  10
      7   5
      3   2
      0   1
     -3   0.5
     -7   0.2
    -10   0.1
    outlier_ratio -- percentage of outlier data points, should be in [0, 1]
    red_noise_ratio -- How large is the variance of the red noise with
    respect to the white noise component. The red noise is not accounted 
    into the uncertainties (s). Setting red_noise_ratio=0.0, would yield
    "perfect" errorbars.
    """
    if outlier_ratio < 0.0 or outlier_ratio > 1.0:
        raise ValueError("Outlier ratio should be in [0 , 1]")
    np.random.seed(rseed)
    N = len(t)
    # First we generate s from a Gamma distribution
    b = 1.0
    k = 3.0
    s = np.random.gamma(shape=k, scale=b, size=(N,))
    mean_ds_squared = k*(1+k)*b**2   # From E[s**2]
    # Draw a heteroscedastic white noise vector
    white_noise = np.random.multivariate_normal(np.zeros(N,), np.diag(s**2))
    # Now we generate a colored noise vector which is unaccounted in s
    #phi = 0.5  # correlation coefficient
    red_noise_variance = mean_ds_squared*red_noise_ratio
    # First order markovian process to generate 
    red_noise = first_order_markov_process(t, red_noise_variance, 10.0)
    """
    # The following is not ok for irregularly sampled time series
    red_noise = np.random.randn(N)*np.sqrt(red_noise_variance)
    for i in range(1, N):
        red_noise[i] = phi*red_noise[i-1] + np.sqrt(1 - phi**2)*red_noise[i]
    """
    # The final noise vector
    noise = white_noise + red_noise 
    var_noise = mean_ds_squared + red_noise_variance
    SNR_unitless = 10.0**(SNR/10.0)
    y = np.sqrt(SNR_unitless*var_noise)*y
    y_noisy = y + noise
    # Add outliers with a certain percentage
    rperm = np.where(np.random.uniform(size=N) < outlier_ratio)[0]    
    outlier = np.random.uniform(5.0*np.std(y), 10.0*np.std(y), size=len(rperm))
    y_noisy[rperm] += outlier
    return y, y_noisy, s
