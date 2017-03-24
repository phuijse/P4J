P4J
===

**Description**

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
Phase Dispersion Minimization [5] and Lafler-Kinman's string length [6]. 

**Contents**

-  Quadratic Mutual Information periodogram for light curves 
-  Phase Dispersion Minimization and String Length methods
-  Basic synthetic light curve generator

**Instalation**

::

    pip install P4J

**Example**

https://github.com/phuijse/P4J/examples/periodogram\_demo.ipynb

**TODO**

-  Multidimensional time series support
-  More period detection criteria (Conditional Entropy)

**Authors**

-  Pablo Huijse pablo.huijse@gmail.com (Millennium Institute of
   Astrophysics and Universidad de Chile)
-  Pavlos Protopapas (Harvard Institute of Applied Computational
   Sciences)
-  Pablo A. Estévez (Millennium Institute of Astrophysics and
   Universidad de Chile)
-  Pablo Zegers (Universidad de los Andes, Chile)
-  José C. Príncipe (University of Florida)

(P4J = Four Pablos and one Jose)

**References**

1. José C. Príncipe, "Information Theoretic Learning: Renyi's Entropy
   and Kernel Perspectives", Springer, 2010
2. Pavlos Protopapas et al., "A Novel, Fully Automated Pipeline for
   Period Estimation in the EROS 2 Data Set", The Astrophysical Journal
   Supplement, vol. 216, n. 2, 2015
3. Pablo Huijse et al., "Computational Intelligence Challenges and
   Applications on Large-Scale Astronomical Time Series Databases", IEEE
   Mag. Computational Intelligence, vol. 9, n. 3, pp. 27-39, 2014
4. Pablo Huijse et al., "An Information Theoretic Algorithm for Finding
   Periodicities in Stellar Light Curves", IEEE Trans. Signal Processing
   vol. 60, n. 10, pp. 5135-5145, 2012
5. R. F. Stellingwerf, "Period determination using phase dispersion 
   minimization", The Astrophysical Journal, vol. 224, pp. 953-960, 1978
6. D. Clarke, "String/Rope length methods using the Lafler-Kinman statistic", 
   Astronomy & Astrophysics, vol. 386, n. 2, pp. 763-774, 2002
