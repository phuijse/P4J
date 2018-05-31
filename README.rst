P4J
===

Description
-----------

P4J is a python package for period detection on irregularly sampled and heteroscedastic time series based on *Information Theoretic* objective functions. P4J was developed for astronomical light curves, irregularly sampled time series of stellar magnitude or flux. The core of this package is a class called periodogram that sweeps an array of periods/frequencies looking for the one that maximizes a given criterion. The main contribution of this work is a criterion for period detection based on the maximization of Cauchy-Schwarz Quadratic Mutual Information (Huijse et al., 2017). Information theoretic criteria incorporate information on the whole probability density function of the process and are more robust than classical second-order statistics based criteria (Principe, 2010). For comparison P4J also incorporates other period detection methods used in astronomy such as the Phase Dispersion Minimization periodogram (Stellingwerf, 1973), Lafler-Kinman's string length (Clarke, 2002) and the Orthogonal multiharmonic AoV periodogram (Schwarzenberg-Czerny, 1996).


Contents
--------

-  Quadratic Mutual Information periodogram for light curves 
-  Phase Dispersion Minimization, String Length, and Analysis of variance periodograms.
-  Basic synthetic light curve generator

Instalation
-----------

Dependencies::

    Numpy
    GCC
    Cython (optional)

If you have a UNIX system the GCC compiler is most likely already installed. If you have a Windows system you may want to install the Microsoft Visual C++ (MSVC) compiler. You can find relevant information at: https://wiki.python.org/moin/WindowsCompilers.

**Note on Cython:** If Cython is found in your system, pyx files are compiled to C sources. If not, the provided C sources are used.

Install from PyPI using::

    pip install P4J

or clone this github and do::

    python setup.py install --user

Example
-------

Please review

    https://github.com/phuijse/P4J/blob/master/examples/periodogram_demo.ipynb

TODO
----

-  Multidimensional time series support
-  More period detection criteria (Conditional Entropy, Lomb-Scargle)
-  Implement block bootstrap for irregular time series

Authors
-------

-  Pablo Huijse pablo.huijse@gmail.com (Millennium Institute of Astrophysics and Universidad Austral de Chile)
-  Pavlos Protopapas (Harvard Institute of Applied Computational Sciences)
-  Pablo A. Estévez (Millennium Institute of Astrophysics and Universidad de Chile)
-  Pablo Zegers (Universidad de los Andes)
-  José C. Príncipe (University of Florida)

(P4J = Four Pablos and one Jose)

Acknowledgment
--------------

We would like to thank the people of the Computational Intelligence laboratory @ UChile, Center for Mathematical Modeling @ Uchile, the Millennium Institute of Astrophysics (www.astrofisicamas.cl), LSST group @ University of Washington and the participants of the Harvard-Chile Data Science school (www.hcds.cl) for their comments and useful discussions. Pablo Huijse acknowledges financial support from FONDECYT through grant 1170305 and postdoctoral grant 3150460, and from the Chilean Ministry of Economy, Development, and Tourism's Millennium Science Initiative through grant IC12009, awarded to The Millennium Institute of Astrophysics, MAS. 


References
----------

1. José C. Príncipe, "Information Theoretic Learning: Renyi's Entropy and Kernel Perspectives", Springer, 2010
2. Pablo Huijse et al., "Robust period estimation using mutual information for multi-band light curves in the synoptic survey era", The Astrophysical Journal Supplement Series, vol. 236, n. 1, 2018, DOI: http://doi.org/10.3847/1538-4365/aab77c, http://arxiv.org/abs/1709.03541
3. Pavlos Protopapas et al., "A Novel, Fully Automated Pipeline for Period Estimation in the EROS 2 Data Set", The Astrophysical Journal Supplement, vol. 216, n. 2, 2015
4. Pablo Huijse et al., "Computational Intelligence Challenges and Applications on Large-Scale Astronomical Time Series Databases", IEEE Mag. Computational Intelligence, vol. 9, n. 3, pp. 27-39, 2014
5. Pablo Huijse et al., "An Information Theoretic Algorithm for Finding Periodicities in Stellar Light Curves", IEEE Trans. Signal Processing vol. 60, n. 10, pp. 5135-5145, 2012
6. Robert F. Stellingwerf, "Period determination using phase dispersion minimization", The Astrophysical Journal, vol. 224, pp. 953-960, 1978, http://adsabs.harvard.edu/abs/1978ApJ...224..953S
7. David Clarke, "String/Rope length methods using the Lafler-Kinman statistic", Astronomy & Astrophysics, vol. 386, n. 2, pp. 763-774, 2002, http://adsabs.harvard.edu/abs/2002A%26A...386..763C
8. Alex Schwarzenberg-Czerny "Fast and Statistically Optimal Period Search in Uneven Sampled Observations", Astrophysical Journal Letters, vol. 460, pp. 107, 1996, http://adsabs.harvard.edu/abs/1996ApJ...460L.107S


