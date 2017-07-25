"""

P4J is a python package for period detection on irregularly sampled and heteroscedastic
time series based on Information Theoretic objective functions. P4J was
developed for astronomical light curves, irregularly sampled time series
of stellar magnitude or flux. The core of this package is a class called periodogram that 
sweeps an array of periods/frequencies looking for the one that maximizes a given criteria. 
The main contribution of this work is a criterion for period detection based on the maximization of
Cauchy-Schwarz Quadratic Mutual Information [1]. Information theoretic criteria incorporate 
information on the whole probability density function of the process and are more robust than 
classical second-order statistics based criteria [2, 3, 4]. For comparison P4J also 
incorporates classical methods for period detection used in astronomy such as
Phase Dispersion Minimization [5], Lafler-Kinman's string length [6], and the 
orthogonal multiharmonic AoV periodogram [7].

Contents:

-  Quadratic Mutual Information periodogram for light curves 
-  Phase Dispersion Minimization, String Length, Orthogonal Multiharmonic AoV
-  Basic synthetic light curve generator


https://github.com/phuijse/P4J

"""

__version__ = '0.27'

from .generator import synthetic_light_curve_generator
from .periodogram import periodogram


