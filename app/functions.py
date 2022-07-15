import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import tensorflow
from scipy.stats import norm
from monte_carlo_class import MonteCarloOptionPricing
import random
import time

ann_1 = keras.models.load_model('ANN-1_h5')
ann_2 = keras.models.load_model('ANN-2_h5')
ann_3 = keras.models.load_model('ANN-3_h5')
ann_4 = keras.models.load_model('ANN-4_h5')

# Black-Scholes Models
def ann_1_pricing(parameters):
    inputs = [np.array([x[0]/x[1], x[2],x[3],x[4]]) for x in [parameters]]
    start_time = time.time()
    predictions_1 = ann_1.predict(np.array([inputs[0]]))
    predictions_1_call = predictions_1[0].flatten()
    predictions_1_call = np.multiply([0 if call < 0 else call for call in predictions_1_call], parameters[1])
    predictions_1_put = predictions_1[1].flatten()
    predictions_1_put = np.multiply([0 if put < 0 else put for put in predictions_1_put], parameters[1])
    ann_1_time = (time.time() - start_time)
    return float(predictions_1_call), float(predictions_1_put), ann_1_time

def ann_2_pricing(parameters):
    inputs = [np.array([x[0]/x[1], x[2],x[3],x[4]]) for x in [parameters]]
    start_time = time.time()
    predictions_2 = ann_2.predict(np.array([inputs[0]]))
    predictions_2_call = predictions_2.flatten()
    predictions_2_call = np.multiply([0 if call < 0 else call for call in predictions_2_call], parameters[1])
    ann_2_time = (time.time() - start_time)
    return float(predictions_2_call), ann_2_time

def ann_3_pricing(parameters):
    inputs = [np.array([x[0]/x[1], x[2],x[3],x[4]]) for x in [parameters]]
    start_time = time.time()
    predictions_3 = ann_3.predict(np.array([inputs[0]]))
    predictions_3_put = predictions_3.flatten()
    predictions_3_put = np.multiply([0 if put < 0 else put for put in predictions_3_put], parameters[1])
    ann_3_time = (time.time() - start_time)
    return float(predictions_3_put), ann_3_time

def black_scholes(parameters):
    """
    Parameters:
        - call_or_put (str): ['call' or 'put']
        - s (float): spot price for underlying
        - k (float): strike price of contract
        - r (float): risk-free interest rate
        - t (float): time to maturity (in years)
        - vol (float): volatility of underlying
    """
    s = parameters[0]
    k = parameters[1]
    r = parameters[2]
    t = parameters[3]
    vol = parameters[4]

    # initialising a normal distribution
    start_time = time.time()
    N = norm.cdf

    d1 = (np.log(s/k) + (r+ vol**2/2)*t) / (vol*np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    call = s * N(d1) - (k * np.exp(-r*t) * N(d2))

    d1 = (np.log(s/k) + (r+ vol**2/2)*t) / (vol*np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    put = (k * np.exp(-r*t) * N(-d2)) - s*N(-d1)
    bs_time = (time.time() - start_time)
    return call, put, bs_time


# Heston MC Models

def ann_4_pricing(parameters):
    inputs = [np.array([x[0]/x[1], x[2],x[3],x[4],x[5],x[6],x[7],x[8]]) for x in [parameters]]
    start_time = time.time()
    predictions_4 = ann_4.predict(np.array([inputs[0]]))
    predictions_4_call = predictions_4[0].flatten()
    predictions_4_call = np.multiply([0 if call < 0 else call for call in predictions_4_call], parameters[1])
    predictions_4_put = predictions_4[1].flatten()
    predictions_4_put = np.multiply([0 if put < 0 else put for put in predictions_4_put], parameters[1])
    ann_4_time = (time.time() - start_time)
    return float(predictions_4_call), float(predictions_4_put), ann_4_time


def heston_monte_carlo(parameters):
    start_time = time.time()
    MC = MonteCarloOptionPricing(S0 = round(parameters[0], 4),
                                 K = round(parameters[1], 4),
                                 T = round(parameters[3], 4),
                                 r = round(parameters[2], 4),
                                 sigma = round(parameters[4], 4),
                                 div_yield = 0,
                                 simulation_rounds = 10000,
                                 no_of_slices = 40,
                                 fix_random_seed = False)
    
    MC.heston(kappa= round(parameters[7], 4),
              theta = round(parameters[6], 4),
              sigma_v = round(parameters[5], 4),
              rho = round(parameters[8], 4))
                                 
    MC.stock_price_simulation()
    
    call_price = MC.european_call()
    put_price = MC.european_put(call_price)
    heston_time = (time.time() - start_time)

    return call_price, put_price, heston_time

# functions for generating random samples
def bs_generation():
    rand = random.randint(0,49999)
    test_inputs = np.genfromtxt('bs_data_for_website.csv', delimiter=',')
    sample = test_inputs[rand]
    rand_spot = sample[0]
    rand_strike = sample[1]
    rand_r = sample[2]
    rand_t = sample[3]
    rand_vol = sample[4]
    return rand_spot, rand_strike, rand_r, rand_t, rand_vol

def heston_generation():
    rand = random.randint(0,24999)
    test_inputs2 = np.genfromtxt('heston_data_for_website.csv', delimiter=',')
    sample = test_inputs2[rand]
    rand_spot = sample[0]
    rand_strike = sample[1]
    rand_r = sample[2]
    rand_t = sample[3]
    rand_vol = sample[4]
    rand_vol_of_vol = sample[5]
    rand_long_average_variance = sample[6]
    rand_reversion_speed = sample[7]
    rand_correlation = sample[8]

    return rand_spot, rand_strike, rand_r, rand_t, rand_vol, rand_vol_of_vol, rand_long_average_variance, rand_reversion_speed, rand_correlation

#print(ann_1_pricing(np.array([2.831159e+02, 2.613451e+02, 6.670000e-02, 2.866000e-01, 6.672000e-01, 5.304180e+01, 2.632250e+01])))
