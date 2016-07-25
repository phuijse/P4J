from __future__ import division, print_function
import numpy as np
from scipy.stats import gamma, exponnorm

class synthetic_light_curve_generator:
    def __init__(self, T, N, rseed=None):
        """
        Class for simple synthetic light curve generation
        
        Light curves are time series of stellar magnitude or flux as a 
        function of time. In the following we consider Earth-based surveys.
        Light curves are irregularly sampled because the measurements are 
        not taken at the same time every night due to observation 
        constraints. Light curves may have data gaps, i.e. periods of time 
        where no observations were registered. The noise present in the 
        light curves comes from the sky background, the Earth atmosphere, 
        telescope systematics, etc.
        
        Parameters
        ----------
        T: float
            Time span of the vector, i.e. how long it is in time
        N: positive integer
            Number of samples of the resulting time vector
        rseed: 
            Random seed to feed the random number generator
        
        """
        self.rseed = rseed
        self.t = irregular_sampling(T, N, rseed)
        self.A = 1.0
        
    def set_model(self, f0, A, model='trigonometric'):
        """
        See also
        --------
        trigonometric_model
        
        """
        self.f0 = f0
        self.y_clean, vary = trigonometric_model(self.t, f0, A)
        # y_clean is normalized to have unit standard deviation
        self.y_clean = self.y_clean/np.sqrt(vary) 
    
    def get_fundamental_frequency(self):
        return self.f0

    def get_clean_signal(self):
        return self.t, self.A*self.y_clean
        
    def draw_noisy_time_series(self, SNR=1.0, red_noise_ratio=0.25, outlier_ratio=0.0):
        """
        A function to draw a noisy time series based on the clean model 
        such that 
            y_noisy = y + yw + yr,
        where yw is white noise, yr is red noise and y will be rescaled 
        so that y_noisy complies with the specified signal-to-noise 
        ratio (SNR). 
        
        Parameters
        ---------
        SNR: float
            Signal-to-noise ratio of the resulting contaminated signal in
            decibels [dB]. SNR is defined as 
                SNR = 10*log(var_signal/var_noise), hence
            NR var_signal/var_noise
                 10  10
                  7   5
                  3   2
                  0   1
                 -3   0.5
                 -7   0.2
                -10   0.1
        red_noise_variance: float in [0, 1]
            The variance of the red noise component is set according to 
            Var(yw)*red_noise_ratio. Set this to zero to obtain uncertainties
            that explain the noise perfectly
        outlier_ratio: float in [0, 1]
            Percentage of outlier data points
            
        Returns
        -------
        t: ndarray
            Vector containing the time instants
        y_noisy: ndarray
            Vector containing the contaminated signal
        s: ndarray
            Vector containing the uncertainties associated to the white 
            noise component

        """
        if outlier_ratio < 0.0 or outlier_ratio > 1.0:
            raise ValueError("Outlier ratio must be in [0, 1]")
        if red_noise_ratio < 0.0:
            raise ValueError("Red noise ratio must be positive")
        np.random.seed(self.rseed)
        t = self.t
        y_clean = self.y_clean
        N = len(t)
        # First we generate s 
        s, mean_s_squared = generate_uncertainties(N, rseed=self.rseed)
        #print(mean_s_squared)
        #print(np.mean(s**2))
        # Draw a heteroscedastic white noise vector
        white_noise = np.random.multivariate_normal(np.zeros(N,), np.diag(s**2))
        # Now we generate a colored noise vector which is unaccounted by s
        red_noise_variance = mean_s_squared*red_noise_ratio
        # First order markovian process to generate 
        red_noise = first_order_markov_process(t, red_noise_variance, 1.0, rseed=self.rseed)
        
        # The following is not ok for irregularly sampled time series because
        # it assumes constant dt=1
        #phi=0.5
        #red_noise = np.random.randn(N)*np.sqrt(red_noise_variance)
        #for i in range(1, N):
        #    red_noise[i] = phi*red_noise[i-1] + np.sqrt(1 - phi**2)*red_noise[i]
        
        # The final noise vector
        #print("%f %f" % (np.var(white_noise)*red_noise_ratio, np.var(red_noise)))
        noise = white_noise + red_noise 
        var_noise = mean_s_squared + red_noise_variance
        SNR_unitless = 10.0**(SNR/10.0)
        self.A = np.sqrt(SNR_unitless*var_noise)
        y = self.A*y_clean
        y_noisy = y + noise
        # Add outliers with a certain percentage
        rperm = np.where(np.random.uniform(size=N) < outlier_ratio)[0]    
        outlier = np.random.uniform(5.0*np.std(y), 10.0*np.std(y), size=len(rperm))
        y_noisy[rperm] += outlier
        return t, y_noisy, s




def irregular_sampling(T, N, rseed=None):
    """
    Generates an irregularly sampled time vector by perturbating a 
    linearly spaced vector and latter deleting a certain number of 
    points
    
    Parameters
    ----------
    T: float
        Time span of the vector, i.e. how long it is in time
    N: positive integer
        Number of samples of the resulting time vector
    rseed: 
        Random seed to feed the random number generator
        
    Returns
    -------
    t_irr: ndarray
        An irregulary sampled time vector
        
    """
    sampling_period = (T/float(N))
    N = int(N)
    np.random.seed(rseed)    
    t = np.linspace(0, T, num=5*N)
    # First we add jitter
    t[1:-1] += sampling_period*0.5*np.random.randn(5*N-2)
    # Then we do a random permutation and keep only N points 
    P = np.random.permutation(5*N)
    t_irr = np.sort(t[P[:N]])
    return t_irr
    

def trigonometric_model(t, f0, A):
    """
    Generates a simple trigonometric model based on a sum of sine waves 
    
    Parameters
    ----------
    t: ndarray
        Vector containing the time instants where the model will be sampled
    f0: float
        Fundamental frequency of the model
    A: ndarray 
        Vector containing the amplitudes of the harmonics in the model. 
        The lenght of A defines the number of harmonics
        
    Returns
    -------
    y: ndarray
        Trigonometric model based
    var_y: float
        Analytical variance of the trigonometric model
        
    Example
    -------
    Assumming a time vector t has already been specified
    
    >>> import P4J
    >>> y_clean = P4J.trigonometric_model(t, f0=2.0, A=[1.0, 0.5, .25])
    
    """
    y = 0.0
    M = len(A)
    var_y = 0.0
    for k in range(0, M):
        y += A[k]*np.sin(2.0*np.pi*t*f0*(k+1))
        var_y += 0.5*A[k]**2
    return y, var_y


def first_order_markov_process(t, variance, time_scale, rseed=None):
    """
    Generates a correlated noise vector using a multivariate normal
    random number generator with zero mean and covariance
    
        Sigma_ij = s^2 exp(-|t_i - t_j|/l),
    
    where s is the variance and l is the time scale. 
    The Power spectral density associated to  this covariance is 
    
        S(f) = 2*l*s^2/(4*pi^2*f^2*l^2 +1),
        
    red noise spectrum is defined as proportional to 1/f^2. This 
    covariance is the one expected from a first order markov process 
    (Reference?)
    
    Parameters
    ---------
    t: ndarray
        A time vector for which the red noise vector will be sampled
    variance: positive float
        variance of the resulting red noise vector
    time_scale: positive float
        Parameter of the covariance matrix
        
    Returns
    -------
    red_noise: ndarray
        Vector containing the red noise realizations
        
    See also
    --------
    power_law_noise
        
    """
    if variance < 0.0:
        raise ValueError("Variance must be positive")
    if time_scale < 0.0:
        raise ValueError("Time scale must be positive")
    np.random.seed(rseed)
    N = len(t)
    mu = np.zeros(shape=(N,))
    if variance == 0.0:
        return mu
    dt = np.repeat(np.reshape(t, (1, -1)), N, axis=0)
    dt = np.absolute(dt - dt.T)  # This is NxN
    S = variance*np.exp(-np.absolute(dt)/time_scale)
    red_noise = np.random.multivariate_normal(mu, S)
    return red_noise


def power_law_noise(t, variance):
    """
    CAUTION: NOT THOROUGHLY TESTED
    Generates a red noise vector by first generating a covariance 
    function based on the expected red noise spectra. Red noise spectrum
    follows a power law
    
        S(f) = A*f^(-2),
    
    where A is a constant. Pink noise can be obtained if the exponent is
    changed to -1, and white noise if it is changed to 0.
    
    This Power Spectral density (PSD) is real and even then
    
        r(tau) = AT sum cos(2pi k tau/T)/k**2,
    
    where we assume f[k] = k*df, df = 1/T, Fs = N/T.
    
    The basel problem gives us: sum (1/k**2) approx pi**2/6, hence we can
    set c according to the desired variance at lag=0
     
    After the covariance is computed we can draw from a multivariate
    normal distribution to obtain a noise vector
    
    Parameters
    ---------
    t: ndarray
        A time vector for which the red noise vector will be sampled
    variance: positive float
        variance of the resulting red noise vector
        
    Returns
    -------
    red_noise: ndarray
        Vector containing the red noise realizations
        
    See also
    --------
    first_order_markov_process
    
    """
    if variance < 0.0:
        raise ValueError("Variance must be positive")
    mu = np.zeros(shape=(N,))
    if variance == 0.0:
        return mu
    N = len(t)    
    T = (t[-1] - t[0])
    c = (6.0*variance)/(T*np.pi**2)
    f = np.arange(1.0/T, 0.5*N/T, step=1.0/T)  # We ommit f=0.0
    k = f*T
    dt = np.repeat(np.reshape(t, (1, -1)), N, axis=0)
    dt = np.absolute(dt - dt.T)  # This is NxN
    S = np.zeros(shape=(N, N))
    for i in range(0, N):
        for j in range(i, N):
            S[i, j] = np.sum(np.cos(2.0*np.pi*k*dt[i, j]/T)/k**2)*T*c
            S[j, i] = S[i, j] 
    red_noise = np.random.multivariate_normal(mu, S)
    return red_noise


def generate_uncertainties(N, dist='Gamma', rseed=None):
    """
    This function generates a uncertainties for the white noise component
    in the synthetic light curve. 
    
    Parameters
    ---------
    N: positive integer
        Lenght of the returned uncertainty vector
    dist: {'EMG', 'Gamma'}
        Probability density function (PDF) used to generate the 
        uncertainties
    rseed:
        Seed for the random number generator
        
    Returns
    -------
    s: ndarray
        Vector containing the uncertainties
    expected_s_2: float
        Expectation of the square of s computed analytically
        
    """
    np.random.seed(rseed)  
    print(dist)
    if dist == 'EMG':  # Exponential modified Gaussian
        # the mean of a EMG rv is mu + 1/(K*sigma)
        # the variance of a EMG rv is sigma**2 + 1/(K*sigma)**2
        K = 1.824328605481941
        sigma = 0.05*0.068768312946785953
        mu = 0.05*0.87452567616276777
        # IMPORTANT NOTE
        # These parameters were obtained after fitting uncertainties
        # coming from 10,000 light curves of the VVV survey
        expected_s_2 = sigma**2 + mu**2 + 2*K*mu*sigma + 2*K**2*sigma**2 
        s = exponnorm.rvs(K, loc=mu, scale=sigma, size=N)
    elif dist == 'Gamma':
        # The mean of a gamma rv is k*sigma
        # The variance of a gamma rv is k*sigma**2
        k = 3.0
        sigma = 0.05/k  #  mean=0.05, var=0.05**2/k
        s = gamma.rvs(k, loc=0.0, scale=sigma, size=N)
        expected_s_2 = k*(1+k)*sigma**2  
    return s, expected_s_2
