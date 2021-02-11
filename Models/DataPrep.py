# -*- coding: utf-8 -*-

"""
Created on Thu 040518
@author: pariya pourmohammadi

pep-8 style
load the data and pre-process it
"""

import numpy as np
import DataType
import os

def patch_maker(num_cols, num_rows, n, x, y):
    """

    Parameters
    ----------
    num_cols : int
        number of columns
    num_rows : int
        number of rows
    n : int
        patch size
    x : array
        model parameters
    y : array
        labeled data for training and test

    Returns
    -------
    x_patches:  4d numpy array
        a 4d array of n by n patches of the model parametrs
    y_patches : 4d numpy array
        a 4d array of n by n patches of the labeled data
    """
    i = 0
    y_patches = np.array(
        [y[i:i + n, j:j + n] for i in range(0, num_rows, n) for j in
         range(0, num_cols, n)])
    x_patches = np.array(
        [x[:, i:i + n, j:j + n] for i in range(0, num_rows, n) for j in
         range(0, num_cols, n)])

    patch_size = x_patches.shape[0]

    while i < patch_size:
        temp = x_patches[i][11, :, :]
        if np.sum(temp) == 0:
            x_patches = np.delete(x_patches, i, 0)
            y_patches = np.delete(y_patches, i, 0)
            patch_size -= 1
            i += 1
        patch_size -= 1
        i += 1

    y_patches = y_patches.reshape(y_patches.shape[0], 1, y_patches.shape[1],
                                  y_patches.shape[2])
    return x_patches, y_patches


## Partition data /since our data is class imbalanced we partitioned the data
# using stratified sampling in this function


def partition_data(train, test, base_train):
    """
    
    Parameters
    ----------
    train : float
        the portion of train data
    test : float
        the portion of test data (validation is data is 1 - (train+test))
    base_train : array
        a sample data of the whole region with the same size as each layer

    Returns
    -------
    dat : 2d array
        an array of 1s (training), 2s(test), and 3s(validation) partitions
    """

    rows = base_train.shape[0]
    cols = base_train.shape[1]

    dat = np.zeros((rows, cols))
    ones = np.where(base_train == 1)
    zeros = np.where(base_train == 0)
    len_ones, len_zeros = np.arange(len(ones[0])), np.arange(len(zeros[0]))

    np.random.shuffle(len_ones)
    np.random.shuffle(len_zeros)

    rng_train_one = len_ones[:int(train * len(ones[0]))]
    dat[ones[0][rng_train_one], ones[1][rng_train_one]] = 1

    rng_test_one = len_ones[int(train * len(ones[0])):int(
        (train + test) * len(ones[0]))]
    dat[ones[0][rng_test_one], ones[1][rng_test_one]] = 2

    rng_val_one = len_ones[int((train + test) * len(ones[0])):]
    dat[ones[0][rng_val_one], ones[1][rng_val_one]] = 3

    rng_train_zeros = len_zeros[:int(train * len(zeros[0]))]
    dat[zeros[0][rng_train_zeros], zeros[1][rng_train_zeros]] = 1

    rng_test_zeros = len_zeros[int(train * len(zeros[0])):int(
        (train + test) * len(zeros[0]))]
    dat[zeros[0][rng_test_zeros], zeros[1][rng_test_zeros]] = 2

    rng_val_zeros = len_zeros[int((train + test) * len(zeros[0])):]
    dat[zeros[0][rng_val_zeros], zeros[1][rng_val_zeros]] = 3

    return dat


# partition data and remove nulls
def split(data, check_dat, partition_dat):
    """

    Parameters
    ----------
    data : 2d array
        array to split

    check_dat : 2d array
        a sample data with null points

    partition_dat : 2d array
        partition of train, test and validation data


    Returns
    -------
    train : vector
        a vector of training data

    test : vector
        a vector of test data

    valid : vector
        a vector of validation data

    """
    n = data.shape[0]*data.shape[1]
    data = data.reshape((n, 1))

    n_train = ((check_dat != -9999.0) & (partition_dat == 1)).sum()
    n_test = ((check_dat != -9999.0) & (partition_dat == 2)).sum()
    n_valid = ((check_dat != -9999.0) & (partition_dat == 3)).sum()

    train = np.zeros((n_train, 1))
    test = np.zeros((n_test, 1))
    valid = np.zeros((n_valid, 1))

    train[:, 0] = data[(check_dat != -9999.0) & (partition_dat == 1)]
    test[:, 0] = data[(check_dat != -9999.0) & (partition_dat == 2)]
    valid[:, 0] = data[(check_dat != -9999.0) & (partition_dat == 3)]

    train = data_train.reshape(train.size)
    test = test.reshape(test.size)
    valid = valid.reshape(valid.size)

    return train, test, valid


def line_extract(inFileName):
    """
    Parameters
    ----------
    inFileName : str
        .txt file created from a shapefile with header

    Returns
    -------
    lines : str
        a string of the first 6 lines in the infile

    """
    file = open(inFileName, 'r')

    all_lines = file.readlines()
    lines = all_lines[0:6][:]

    return lines


# Load and Pre-process Data
def load(code_path, data_path):
    """
    Parameters
    ----------
    code_path : str
        the path to the codes directory

    data_path : str
        the path to the data directory

    Returns
    -------
        No returns, saves the data of training test and validation as .npy
    """

    os.chdir(code_path)

    os.chdir(data_path)
    lines = line_extract('ag_den.txt')
    number_cols = int(lines[0][14:18])
    number_rows = int(lines[1][14:18])
    total_size = number_rows * number_cols

    land_use_base = np.genfromtxt('landuse_base.txt',
                                  delimiter=' ',
                                  skip_header=6)
    land_use_base_var = DataType.get_var_names(land_use_base)

    partition_dat = partition_data(0.60,
                                   0.2,
                                   land_use_base)
    np.array(partition_dat)
    partition_dat = partition_dat.reshape(total_size, 1)

    check_dat = np.genfromtxt('ag_den.txt', delimiter=' ', skip_header=6)
    np.array(check_dat)
    check_dat = check_dat.reshape(total_size, 1)

    n_train = ((check_dat != -9999.0) & (partition_dat == 1)).sum()
    n_test = ((check_dat != -9999.0) & (partition_dat == 2)).sum()
    n_valid = ((check_dat != -9999.0) & (partition_dat == 3)).sum()


    maj_1000 = np.genfromtxt('maj_1000.txt',
                             delimiter=' ',
                             skip_header=6)

    maj_1000_var = DataType.get_var_names(maj_1000)

    maj_100 = np.genfromtxt('maj_100.txt', delimiter=' ', skip_header=6)
    maj_100_var = DataType.get_var_names(maj_100)

    # take the length of each vaiable's names and add it to them up
    final_len = 13 + len(land_use_base_var) + len(maj_1000_var) + len(
        maj_100_var) + - 3

    dat_train = np.zeros((n_train, final_len))

    dat_test = np.zeros((n_test, final_len))

    data_validation = np.zeros((n_valid, final_len))

    a: int = 0

    ag_den = np.genfromtxt('ag_den.txt', delimiter=' ', skip_header=6)
    ag_den = DataType.normalize(ag_den)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(ag_den, check_dat, partition_dat)
    a += 1

    aspect = np.genfromtxt('aspect.txt', delimiter=' ', skip_header=6)
    aspect = DataType.normalize(aspect)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(aspect, check_dat, partition_dat)
    a += 1

    dev_den = np.genfromtxt('dev_den.txt', delimiter=' ', skip_header=6)
    dev_den = DataType.normalize(dev_den)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(dev_den, check_dat, partition_dat)
    a += 1

    dist_dev = np.genfromtxt('dist_dev.txt', delimiter=' ', skip_header=6)
    dist_dev = DataType.normalize(dist_dev)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(dist_dev, check_dat, partition_dat)
    a += 1

    dist_mine = np.genfromtxt('dist_mines.txt', delimiter=' ', skip_header=6)
    dist_mine = DataType.normalize(dist_mine)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(dist_mine, check_dat, partition_dat)
    a += 1

    dist_pascrop = np.genfromtxt('dist_pastcrop.txt', delimiter=' ',
                                 skip_header=6)
    dist_pascrop = DataType.normalize(dist_pascrop)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(dist_pascrop, check_dat, partition_dat)
    a += 1

    dist_rec = np.genfromtxt('dist_rec.txt', delimiter=' ', skip_header=6)
    dist_rec = DataType.normalize(dist_rec)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(dist_rec, check_dat, partition_dat)
    a += 1

    dist_road = np.genfromtxt('dist_road.txt', delimiter=' ', skip_header=6)
    dist_road = DataType.normalize(dist_road)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(dist_road, check_dat, partition_dat)
    a += 1

    dist_stream = np.genfromtxt('dist_stream.txt', delimiter=' ',
                                skip_header=6)
    dist_stream = DataType.normalize(dist_stream)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(dist_stream, check_dat, partition_dat)
    a += 1

    elev = np.genfromtxt('elev.txt', delimiter=' ', skip_header=6)
    elev = DataType.normalize(elev)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(elev, check_dat, partition_dat)
    a += 1

    pop_2000 = np.genfromtxt('pop_2000.txt', delimiter=' ', skip_header=6)
    pop_2000 = DataType.normalize(pop_2000)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(pop_2000, check_dat, partition_dat)
    a += 1

    slope = np.genfromtxt('slope.txt', delimiter=' ', skip_header=6)
    slope = DataType.normalize(slope)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(slope, check_dat, partition_dat)
    a += 1

    oil_gas_den_kernel = np.genfromtxt('og_den.txt', delimiter=' ',
                                       skip_header=6)
    oil_gas_den_kernel = DataType.normalize(oil_gas_den_kernel)
    dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
        split(oil_gas_den_kernel, check_dat, partition_dat,)
    a += 1

    cat_file = open(data_path + '/categories.txt', 'w')

    land_use_base = np.genfromtxt('landuse_base.txt', delimiter=' ',
                                  skip_header=6)
    land_use_base_var = DataType.get_var_names(land_use_base)
    if len(land_use_base_var) > 1:
        land_use_base = DataType.get_dummy(land_use_base, land_use_base_var[1])
        dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
            split(land_use_base, check_dat, partition_dat)
        a += 1

    else:
        pass
    cat_file.write('landuse_base of this region has ' +
                   str(land_use_base_var) +
                   ' classes\n')

    state = np.genfromtxt('state.txt', delimiter=' ')
    state_var = DataType.get_var_names(state)
    state_dummy = {}
    if len(state_var) > 1:
        for count in range(1, len(state_var)):
            state_dummy['%s' % count] = DataType.get_dummy(state,
                                                           state_var[count])

            dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
                split(state_dummy['%s' % count]
                      , check_dat
                      , partition_dat)

            a += 1
            print(a)
        a = a + len(state_var) - 1
    cat_file.write('state of this region has '
                   + str(len(state_var))
                   + ' classes\n')

    counties = np.genfromtxt('counties.txt', delimiter=' ')
    counties_var = DataType.get_var_names(counties)
    if len(counties_var) > 1:
        counties_dummy = {}
        for count in range(1, len(counties_var)):
            counties_dummy['%s' % count] = \
                DataType.get_dummy(counties, counties_var[count])
            dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
                split(counties_dummy['%s' % count]
                      , check_dat
                      , partition_dat)
            a += 1
            print(a)
        cat_file.write('counties of this region has '
                       + str(len(counties_var))
                       + ' classes\n')
        print(a)

        lc2001 = np.genfromtxt('lc2001.txt', delimiter=' ')
        lc2001_var = DataType.get_var_names(lc2001)
        lc_2001_dummy = {}
        lc2001_names = []
        if len(lc2001_var) > 1:
            for t in range(1, len(lc2001_var)):
                lc_2001_dummy['%s' % t] = \
                    DataType.get_dummy(lc2001, lc2001_var[t])
                dat_train[:, a], dat_test[:, a], data_validation[:, a] = \
                    split(lc_2001_dummy['%s' % t]
                          , check_dat
                          , partition_dat)
                a += 1
                print(a)
            print(a)

        cat_file.write('land cover 2001 of this region has '
                       + str(len(lc2001_names))
                       + ' classes\n')

    maj_1000 = np.genfromtxt('maj_1000.txt'
                             , delimiter=' '
                             , skip_header=6)
    maj_1000_var = DataType.get_var_names(maj_1000)
    if len(maj_1000_var) > 1:
        maj_1000_dummy = {}
        for t in range(1, len(maj_1000_var)):
            maj_1000_dummy['%s' % t] = DataType.get_dummy(maj_1000,
                                                          maj_1000_var[t])
            data_train[:, a], data_test[:, a], data_val[:, a] = \
                split(maj_1000_dummy['%s' % t], check_dat, partition_dat)
            a += 1
            print(a)
    cat_file.write(
        'major classes of land within 1000 meters of this region has ' + str(
            len(maj_1000_var)) + ' classes\n')
    print(a)

    maj_100 = np.genfromtxt('maj_100.txt', delimiter=' ', skip_header=6)
    maj_100_var = DataType.get_var_names(maj_100)
    if len(maj_100_var) > 1:
        maj_100_dummy = {}
        for t in range(1, len(maj_100_var)):
            maj_100_dummy["%s" % t] = DataType.get_dummy(maj_100,
                                                         maj_100_var[t])
            data_train[:, a], data_test[:, a], data_val[:, a] = \
                split(maj_100_dummy['%s' % t], check_dat, partition_dat)
            a += 1
            print(a)

    cat_file.write(
        'major classes of land within 100 meters of this region has ' + str(
            len(maj_100_var)) + ' classes\n')

    cat_file.close()

    land_use_final = np.genfromtxt('land_use_final.txt', delimiter=' ',
                                   skip_header=6)
    land_use_final_var = DataType.get_var_names(land_use_final)
    land_use_final = DataType.get_dummy(land_use_final, land_use_final_var[0])
    label_train, label_test, label_validation = \
        split(land_use_final, check_dat, partition_dat)

    np.save("partition", partition_dat)
    np.save("check", check_dat)

    np.save("data_train", data_train)
    np.save("data_test", data_test)

    np.save("label_train", label_train)
    np.save("label_valid", label_validation)
    np.save("label_test", label_test)




## Convert the result back to txt file and put each cell
# at its specific location for Visualization
def convert_to_output(y_out, data_path, file):
    """

    Parameters
    ----------
    y_out : 2d array
        model result for the test data

    data_path : str
        the directory at which the data should be saved

    file : str
        .txt file and the directory of a labeled data

    Returns
    -------
    y : 2d array
        an array of the results and y train and validation
    """

    results = np.genfromtxt(file, delimiter=' ',
                            skip_header=6)

    number_rows = results.shape[0]
    number_cols = results.shape[1]

    np.array(results)
    results = results.reshape((number_rows*number_cols, 1))

    partition_ref = np.load('partition.npy')
    data_train = np.load('data_train.npy')
    data_val = np.load('data_val.npy')

    y = results
    y[(results != -9999.0) & (partition_ref == 1)] = data_train[:, 0]
    y[(results != -9999.0) & (partition_ref == 2)] = y_out[:, 0]
    y[(results != -9999.0) & (partition_ref == 3)] = data_val[:, 0]
    y = y.reshape((number_rows, number_cols))
    np.savetxt(data_path + '/y_predict.txt', y, fmt='%1.1f', delimiter=' ')

    return y
