# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:27:27 2018

@author: pariya pourmohammadi
"""

import numpy as np


def partition_data(rows, cols):
    """

    Parameters
    ----------
    rows :
        number of rows
    cols :
        number of columns
    Returns
    -------
    dat : array
        the random partition array
    """
    dat = np.random.random((rows, cols))
    dat[dat >= 0.4] = 1
    dat[(dat >= 0.2) & (dat < 0.4)] = 2
    dat[dat < 0.2] = 3

    return dat
