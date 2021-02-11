# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 05:00:00 2017

@author: pariya pourmohammadi
"""
import numpy as np
import heapq


# get the list of variable names
def get_var_names(data):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """
    var_list = list(set(np.asarray(data).reshape(-1)))
    var_list = sorted(np.array(var_list))
    var_list = var_list[1:]
    return var_list


# Normalize a 2d array of continuous  data between 0 and 1
def normalize(data):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """
    min_val = heapq.nsmallest(2, set(np.asarray(data).reshape(-1)))[-1]
    data = np.add(data, -min_val)
    temp = np.max(data) - min_val
    data = np.divide(data, temp)
    mask = (data < 0)
    data[mask] = 0
    return data


# Create dummy list from a data based on var name
def get_dummy(data, var):
    """

    Parameters
    ----------
    data
    var

    Returns
    -------

    """
    dim = np.shape(data)
    mask = (data == var)
    new_array = np.zeros(dim)
    new_array[mask] = 1
    return new_array
