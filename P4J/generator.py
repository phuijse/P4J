import numpy as np

"""
Generates an irregularly sampled time vector with specified number of 
samples (N) and time span (T)
"""
def irregular_sampling(T, N, rseed=None):
    N = int(N)
    np.random.seed(rseed)
    sampling_period = (T/N)
    t = np.linspace(0, T, num=5*N)
    # First we add jitter
    t[1:-1] += sampling_period*0.5*np.random.randn(5*N-2)
    # Then we do a random permutation and keep only N points 
    P = np.random.permutation(5*N)
    return np.sort(t[P[:N]])
    
"""
Generates a normalized trigonometric model, A corresponds to a vector
having the amplitudes of the model and t is a time vector, f0 is the 
fundamental frequency of the model
"""
def trigonometric_model(t, f0, A):
    y = 0.0
    M = len(A)
    var_y = 0.0
    for k in range(0, M):
        y += A[k]*np.sin(2.0*np.pi*t*f0*(k+1))
        var_y += 0.5*A[k]**2
    return y/np.sqrt(var_y)

"""
A function that takes an irregulary sampled time series (t, y) 
and contaminates it, simulating the noise present in light curves. 
It returns (y, y_noisy, s), where s are the uncertainties 
associated to the magnitudes (y_noisy). y will be reescaled to fit a 
given SNR. SNR is the Signal to Noise ratio of the resulting time series 
and out_p is the percentage of outlier data points. 
SNR = var_signal**2/var_noise**2
"""
def contaminate_time_series(t, y, SNR=10.0, out_p=0.01, rseed=None):
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
    white_over_red_var = 2.0 
    phi = 0.5  # correlation coefficient
    red_noise_variance = mean_ds_squared/white_over_red_var
    red_noise = np.random.randn(N)*np.sqrt(red_noise_variance)
    for i in range(1, N):
        red_noise[i] = phi*red_noise[i-1] + np.sqrt(1 - phi**2)*red_noise[i]
    # The final noise vector
    noise = white_noise + red_noise 
    var_noise = mean_ds_squared + red_noise_variance
    y = np.sqrt(SNR*var_noise)*y
    y_noisy = y + noise
    # Add outliers with a certain percentage
    rperm = np.where(np.random.uniform(size=N) < out_p)[0]    
    outlier = np.random.uniform(5.0*np.std(y), 10.0*np.std(y), size=len(rperm))
    y_noisy[rperm] += outlier
    return y, y_noisy, s
