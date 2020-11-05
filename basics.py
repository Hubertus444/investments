import numpy as np
import numpy.linalg as linalg
import math
import random

# utils

def is_row_vector(vector):
    return vector.shape[0] == 1


def is_col_vector(vector):
    return vector.shape[1] == 1


def matrix_dimensions_match(first, second):
    return first.shape[1] == second.shape[0]

# functions

def portfolio_returns(portfolio_weights, expected_returns):
    '''
    input does not need to be transposed
    '''
    if not is_row_vector(portfolio_weights):
        portfolio_weights = np.transpose(portfolio_weights)

    if not is_col_vector(expected_returns):
        expected_returns = np.transpose(expected_returns)

    if not matrix_dimensions_match(portfolio_weights, expected_returns):
        print("ERROR: Your dimensions do not match!")
        return None

    return np.matmul(portfolio_weights, expected_returns)


def portfolio_variance(portfolio_weights, var_covar_matrix):
    '''
    input does not need to be transposed
    '''
    if not is_row_vector(portfolio_weights):
        portfolio_weights = np.transpose(portfolio_weights)
    if not matrix_dimensions_match(portfolio_weights, var_covar_matrix):
        print("ERROR: Your dimensions do not match!")
        return None    
    return np.matmul(np.matmul(portfolio_weights, var_covar_matrix), np.transpose(portfolio_weights))


def min_var_portfolio(var_covar_matrix):
    '''
    takes a variance-covariance-matrix and
    returns the minimum variance portfolio
    '''
    ones_vector = np.ones((var_covar_matrix.shape[0],1))
    mat_sum_axis1 = np.matmul(linalg.inv(var_covar_matrix), ones_vector)
    return mat_sum_axis1 / np.matmul(np.transpose(ones_vector), mat_sum_axis1)


def find_min_var_portfolio_for_fixed(fixed_asset, other_assets, var_covar_matrix, repetitions=100000):
    variance = math.inf
    for _ in range(repetitions):
        fa = [fixed_asset]
        num = 1000
        sec = random.randrange(-600,600)
        while True:
            tri = random.randrange(-600,600)
            if (400 + sec + tri) <= 1000:
                break
        fo = 600 - tri - sec
        fa = [0.4, sec / num, tri / num, fo / num]
        w = np.array([fa])
        new_variance = portfolio_variance(w, var_covar_matrix)
        if new_variance < variance:
            variance = new_variance
            portfolio_weights = w

    return portfolio_weights, variance


def find_min_var_portfolio_for_return(goal_return, expected_returns, var_covar_matrix, repetitions=100000, no_short_sales=False):
    variance = math.inf
    for i in range(repetitions):
        if no_short_sales:
            w = np.zeros((4,1))
            for i in range(w.shape[0]):
                w[i] = random.random()
        else:
            w = np.zeros((4,1))
            for i in range(w.shape[0]):
                w[i] = random.randrange(-1000,1000)
        if np.sum(w) == 0:
            continue
        w /= np.sum(w)
        new_variance = portfolio_variance(w, var_covar_matrix)
        new_expec_return = portfolio_returns(w, expected_returns)
        if new_variance < variance and abs(goal_return - np.average(new_expec_return)) < 0.01:
            variance = new_variance
            goal_return = new_expec_return
            portfolio_weights = w

    return portfolio_weights, goal_return, variance


def find_min_var_portfolio_for_var(goal_var, expected_returns, var_covar_matrix, repetitions=100000, no_short_sales=False):
    expected_r = - math.inf
    for i in range(repetitions):
        if no_short_sales:
            w = np.zeros((4,1))
            for i in range(w.shape[0]):
                w[i] = random.random()
        else:
            w = np.zeros((4,1))
            for i in range(w.shape[0]):
                w[i] = random.randrange(-1000,1000)
        if np.sum(w) == 0:
            continue
        w /= np.sum(w)
        new_variance = portfolio_variance(w, var_covar_matrix)
        new_expec_return = portfolio_returns(w, expected_returns)
        if new_expec_return > expected_r and abs(goal_var - new_variance) < 0.1:
            variance = new_variance
            expected_r = new_expec_return
            portfolio_weights = w

    return portfolio_weights, expected_r, variance


