#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

"""
    Expands an array with values x following the counts
    x is a N float array
    counts is a N integer array
"""


def weight_array(x, counts):
    zipped = zip(x, counts)
    weighted = []
    for i in zipped:
        for j in range(i[1]):
            weighted.append(i[0])
    return weighted

"""
    Computes the weighted interquartile range (IQR)
    x is a N float array
    weights is a N float array, it expects sum w_i = 1
"""


def wIQR(x, weights):

    wx = weight_array(x, np.round(2.0*weights/np.amin(weights)).astype('int'))
    return np.percentile(wx, 75) - np.percentile(wx, 25)

"""
    Computes the weighted standard deviation
    x is a N float array
    weights is a N float array, it expects sum w_i = 1
"""
def wSTD(x, weights):
    wmean = np.average(x, weights=weights)
    return np.sqrt(np.average((x - wmean)**2, weights=weights))

"""
    Computes a robust measure of scale by compare the weighted versions of the standard deviation and the interquartile range
    x is a N float array
    weights is a N float array, it expects sum w_i = 1
"""

def robust_center(x, weights):
    wx = weight_array(x, np.round(2.0*weights/np.amin(weights)).astype('int'))
    return np.percentile(wx, 50)

def robust_scale(x, weights):
    return np.amin([wSTD(x, weights), wIQR(x, weights)/1.349])
