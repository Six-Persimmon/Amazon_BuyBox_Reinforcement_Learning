import random
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import scipy.special as special
from scipy.optimize import fsolve, minimize, root, least_squares
from scipy import integrate, LowLevelCallable
import numba
from numba import cfunc, carray
from numba.types import intc, CPointer, float64
import matplotlib.pyplot as plt
import copy
import multiprocessing as mp
import itertools
import pickle
import math


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def jit_integrand_function(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx, n)
        return jitted_function(values)
    return LowLevelCallable(wrapped.ctypes)

def gen_correct_demand(price, a0, a, mu, ps):
    num_product = len(price)
    a = np.array(a)
    price = np.array(price)
    demand = np.zeros(num_product)
    for i in range(num_product):
        max_logit = np.exp(a0/mu) + np.sum(np.exp((a - ps * price)/mu)) - np.exp((a[i] - ps * price[i])/mu)
        loc_tmp = mu * np.log(max_logit) - a[i] + ps * price[i]
        x_tmp = (0 - loc_tmp) / mu
        demand_tmp = 1/(1 + np.exp(-x_tmp))
        demand[i] = demand_tmp
    return demand

def gen_per_profit_correct(price, a0, a, mu, ps, mc):
    p = np.array(price)
    mc = np.array(mc)
    demand = gen_correct_demand(price, a0, a, mu, ps)
    profit = demand * (p - mc)
    return profit

def function(b, sc, mu, position = 2):
    return  - mu * b + mu * np.euler_gamma - mu * special.expi(-np.exp(-b)) - (position - 1) * sc

@jit_integrand_function
def pdf_joint_2(args):
    b, a0, a_b, p_b, a_u, p_u, mu, ps = args
    x1 = b - (mu * np.log(np.exp(a0/mu) + np.exp((a_b - ps * p_b)/mu)) + ps * p_u - a_u)/mu
    x2 = (mu * b + a_u - ps * p_u - (a_u - ps * p_u))/mu
    return np.exp(-(x1 + np.exp(-x1))) * (1 - np.exp(- np.exp(-x2)))
    
@jit_integrand_function
def pdf_joint_3(args):
    u2, mu, a0, a1, a2, p1, p2, ps = args
    x1 = (u2 - (a2 - ps * p2))/mu
    x2 = (u2 - a0)/mu
    x3 = (u2 - (a1 - ps * p1))/mu
    return np.exp(-(x1 + np.exp(-x1)))/mu * np.exp(-np.exp(-x2)) * (1 - np.exp(-np.exp(-x3)))

def gen_incorrect_demand(a0, p1, p2, a1, a2, mu, sc, ps, b_bound):
    d1 = integrate.quad(pdf_joint_2, -np.inf, b_bound, args = (a0, a2, p2, a1, p1, mu, ps,))
    d2 = integrate.quad(pdf_joint_3, mu * b_bound + a1 - ps * p1, np.inf, args = (mu, a0, a1, a2, p1, p2, ps,))
    return d1, d2

def gen_profit_per_inaccurate_rank(price, a0, a, mc, lamb, mu, sc, ps):
    p1, p2 = price
    a1, a2 = a
    mc1, mc2 = mc
    
    b_bound = fsolve(function, [1], args = (sc, mu))
    part1_1, part1_2 = gen_correct_demand(price, a0, a, mu, ps)
    part2_1, part2_2 = gen_incorrect_demand(a0, p1, p2, a1, a2, mu, sc, ps, b_bound)
    part3_2, part3_1 = gen_incorrect_demand(a0, p2, p1, a2, a1, mu, sc, ps, b_bound)
    d1 = lamb * part1_1 + (1 - lamb) * (part2_1[0] + part3_1[0])
    d2 = lamb * part1_2 + (1 - lamb) * (part2_2[0] + part3_2[0])
    reward1 = (p1 - mc1) * d1
    reward2 = (p2 - mc2) * d2
    return [reward1, reward2]

def gen_demand_unper_two_product(price, a0, a, mu, ps, sc, state):
    '''
    state = 1: firm 1 (low quality firm) is ranked on the first
    state = 0: firm 2 (high quality firm) is ranked on the first
    '''
    p1, p2 = price
    a1, a2 = a
    
    b_bound = fsolve(function, [1], args = (sc, mu))
    d1_state1_correct, d2_state2_correct = gen_correct_demand(price, a0, a, mu, ps)
    d1_state2_incorrect, d2_state2_incorrect = gen_incorrect_demand(a0, p1, p2, a1, a2, mu, sc, ps, b_bound)
    d2_state1_incorrect, d1_state1_incorrect = gen_incorrect_demand(a0, p2, p1, a2, a1, mu, sc, ps, b_bound)
    if state == 1:
        d1 = d1_state1_correct + d1_state1_incorrect
        d2 = d2_state1_incorrect
    elif state == 0:
        d1 = d1_state2_incorrect
        d2 = d2_state2_correct + d2_state2_incorrect
    return d1[0], d2[0]

def gen_profit_unper_two_product(price, a0, a, mc, mu, ps, sc, state):
    p1, p2 = price
    mc1, mc2 = mc
    
    d1, d2 = gen_demand_unper_two_product(price, a0, a, mu, ps, sc, state)
    profit1 = d1 * (p1 - mc1)
    profit2 = d2 * (p2 - mc2)
    return [profit1, profit2]

def gen_unper_ranking_prob(a, price, gamma, score_type = "expected-utility", args = None):
    a = np.array(a)
    price = np.array(price)
    if score_type == "expected-utility":
        rank_score = a + gamma * price
    elif score_type == "popularity":
        rank_score0, _ = gen_demand_unper_two_product(price, args.a0, a, args.mu, args.ps, args.sc, 1)
        _, rank_score1 = gen_demand_unper_two_product(price, args.a0, a, args.mu, args.ps, args.sc, 0)
        rank_score = [rank_score0, rank_score1]
    rank_prob = 1. * (rank_score[0] >= rank_score[1])
    rank_prob = np.array([rank_prob, 1 - rank_prob])
    return rank_prob

def gen_sales_ranking(performance, kappa = 1):
    rank_prob = 1/kappa * np.array(performance)
    rank_prob = np.exp(rank_prob)/np.sum(np.exp(rank_prob))
    state = np.random.choice([1, 0], p = rank_prob)
    return state, rank_prob

def plot_sales_ranking():
    sale = np.linspace(0, 1, 100)
    sales = np.array([sale, 1 - sale])
    rank_prob = np.zeros((2, 100))
    for i in range(100):
        state, rank_prob[:, i] = gen_sales_ranking(sales[:, i])

    fig, ax = plt.subplots()
    plt.plot(sale, rank_prob[0], label = "Low quality firm")
    plt.plot(sale, sale, linestyle = '--', color = 'black')
    plt.show()

def gen_expected_profit_unper_two_product(price, args):
    '''
    state = 1: low quality firm ranks on the top
    state = 0: high quality firm ranks on the top
    '''
    rank_prob = gen_unper_ranking_prob(args.a, price, args.gamma, args.score_type, args=args)
    profit1 = gen_profit_unper_two_product(price, args.a0, args.a, args.mc, args.mu, args.ps, args.sc, 1)
    profit2 = gen_profit_unper_two_product(price, args.a0, args.a, args.mc, args.mu, args.ps, args.sc, 0)
    expected_profit1 = rank_prob[0] * np.array(profit1)
    expected_profit2 = rank_prob[1] * np.array(profit2)
    expected_profit = expected_profit1 + expected_profit2
    return rank_prob, expected_profit.tolist()

def gen_correct_derivative(price, a0, a, mu, ps, mc):
    mc = np.array(mc)
    price = np.array(price)
    demand = gen_correct_demand(price, a0, a, mu, ps)
    derivative = demand + demand * (demand - 1) * (price - mc) * ps/mu
    return derivative

def gen_per_joint_profit(price, mc, a0, a, mu, ps):
    demand = gen_correct_demand(price, a0, a, mu, ps)
    price = np.array(price)
    mc = np.array(mc)
    return np.sum(- (price - mc) * demand)

def gen_per_price_range(a0, a, mc, ps, mu):
    num_product = len(a)
    price0 = 1. * np.ones(num_product)
    res = minimize(gen_per_joint_profit, price0, method='nelder-mead',
                   args = (mc, a0, a, mu, ps),
                   options={'xatol': 1e-8})
    mono_prices = res.x
    print(f"Monopoly prices are {mono_prices}")
    
    while True:
        sol = root(gen_correct_derivative, price0,
                   args = (a0, a, mu, ps, mc))
        comp_prices = sol.x
        if np.all(comp_prices >= np.array(mc)) and np.all(comp_prices <= np.array(a)):
            print(f"Competitive prices are {comp_prices}, finished.")
            break
        else:
            print(f"Competitive prices are {comp_prices}, keep iterating")
            price0 = [random.uniform(mc[i], a[i]) for i in range(num_product)]
    
    comp_profit = gen_per_profit_correct(comp_prices, a0, a, mu, ps, mc)
    mono_profit = gen_per_joint_profit(mono_prices, mc, a0, a, mu, ps)
    return comp_prices, mono_prices, comp_profit, -mono_profit

def obj_func(p, deg_revenue_matrix):
    '''
    p is a permutation vector that shares the same dimension with the price space
    '''
    result = p @ deg_revenue_matrix[:, 0] - p @ deg_revenue_matrix[:, 1:]    
    result = result.tolist() + [sum(p) - 1.]
    return result

def gen_price_range_unpersonalized2(a0, a, mc, mu, ps, gamma, sc, step_size = 0.01, score_type = None, args = None):
    price_space1 = np.arange(mc[0], a[0] + 1.0, step_size)
    price_space1 = np.around(price_space1, 3)
    m1 = len(price_space1)
    price_space2 = np.arange(mc[1], a[1] + 1.0, step_size)
    price_space2 = np.around(price_space2, 3)
    m2 = len(price_space2)
    revenue_matrix1 = np.zeros((m1, m2))
    revenue_matrix2 = np.zeros((m1, m2))
    mon_revenue_matrix1 = np.zeros((m1, m2))
    mon_revenue_matrix2 = np.zeros((m1, m2))
    
    for i in range(m1):
        p1 = price_space1[i]
        for j in range(m2):
            p2 = price_space2[j]
            _, profit = gen_expected_profit_unper_two_product([p1, p2], args)
            revenue_matrix1[i, j] = profit[0]
            revenue_matrix2[i, j] = profit[1]
            profit1 = gen_profit_unper_two_product([p1, p2], a0, a, mc, mu, ps, sc, 1)
            profit2 = gen_profit_unper_two_product([p1, p2], a0, a, mc, mu, ps, sc, 0)
            mon_revenue_matrix1[i, j] = profit1[0]
            mon_revenue_matrix2[i, j] = profit2[1]
    
    flag = True
    tmp_revenue_matrix1 = revenue_matrix1
    tmp_revenue_matrix2 = revenue_matrix2
    old_tmp_revenue_matrix1 = copy.deepcopy(tmp_revenue_matrix1)
    old_tmp_revenue_matrix2 = copy.deepcopy(tmp_revenue_matrix2)
    mask1_list = []
    mask2_list = []
    while flag:
        ind_dominant_revenue_matrix1 = (np.max(tmp_revenue_matrix1, axis = 0, keepdims = True) - tmp_revenue_matrix1) > 0
        mask1 = np.sum(ind_dominant_revenue_matrix1, axis = 1) == tmp_revenue_matrix1.shape[1]
        ind_dominant_revenue_matrix2 = (np.max(tmp_revenue_matrix2, axis = 1, keepdims = True) - tmp_revenue_matrix2) > 0
        mask2 = np.sum(ind_dominant_revenue_matrix2, axis = 0) == tmp_revenue_matrix2.shape[0]
        tmp_revenue_matrix1 = tmp_revenue_matrix1[~mask1, :][:, ~mask2]
        tmp_revenue_matrix2 = tmp_revenue_matrix2[:, ~mask2][~mask1, :]
        mask1_list.append(mask1)
        mask2_list.append(mask2)
        flag_num = (old_tmp_revenue_matrix1.tolist() == tmp_revenue_matrix1.tolist()) + (old_tmp_revenue_matrix2.tolist() == tmp_revenue_matrix2.tolist())
        if flag_num == 2:
            flag = False
        old_tmp_revenue_matrix1 = copy.deepcopy(tmp_revenue_matrix1)
        old_tmp_revenue_matrix2 = copy.deepcopy(tmp_revenue_matrix2)
    
    # calculate mixed strategy for player 1
    m2 = len(tmp_revenue_matrix2)
    p1_init = np.array([1/m2] * m2)
    bounds1 = [[0] * m2, [1] * m2]
    p1 = least_squares(obj_func, p1_init, bounds = bounds1, args = (tmp_revenue_matrix2,))
    p1 = p1.x
    deg_price_space1 = copy.deepcopy(price_space1[~mask1_list[0]])
    for i in range(len(mask1_list) - 1):
        mask1 = mask1_list[i + 1]
        deg_price_space1 = copy.deepcopy(deg_price_space1[~mask1])
    # calculate mixed strategy for player 2
    m1 = len(tmp_revenue_matrix1.T)
    p2_init = np.array([1/m1] * m1)
    bounds2 = [[0] * m1, [1] * m1]
    p2 = least_squares(obj_func, p2_init, bounds = bounds2, args = (tmp_revenue_matrix1.T,))
    p2 = p2.x
    deg_price_space2 = copy.deepcopy(price_space2[~mask2_list[0]])
    for i in range(len(mask2_list) - 1):
        mask2 = mask2_list[i + 1]
        deg_price_space2 = copy.deepcopy(deg_price_space2[~mask2])
    comp_prices = [min(deg_price_space1), min(deg_price_space2)]
    # calculate monopoly price for player 1
    mon1 = max(price_space1[np.argmax(mon_revenue_matrix1, axis = 0)])
    mon2 = max(price_space2[np.argmax(mon_revenue_matrix2, axis = 1)])
    mono_prices = [mon1, mon2]
    # competitive revenue is
    comp_reward2 = p1 @ tmp_revenue_matrix2
    comp_reward1 = p2 @ tmp_revenue_matrix1.T
    comp_profit = [np.mean(comp_reward1), np.mean(comp_reward2)]
    revenue_matrix = revenue_matrix1 + revenue_matrix2
    mono_reward = np.max(revenue_matrix)
    mono_profit = mono_reward
    return comp_prices, mono_prices, comp_profit, mono_profit

def obj_func_ls(p, payoff_matrix_player1, payoff_matrix_player2, payoff_matrix_player3):
    dim0, dim1, dim2 = payoff_matrix_player1.shape
    p1 = p[:dim0]
    p2 = p[dim0 : dim0 + dim1]
    p3 = p[dim0 + dim1:]
    # p1 = np.exp(p1)/np.sum(np.exp(p1))
    # p2 = np.exp(p2)/np.sum(np.exp(p2))
    # p3 = np.exp(p3)/np.sum(np.exp(p3))

    prob_space23 = p2[:, None] * p3[None, :]
    prob_space13 = p1[:, None] * p3[None, :]
    prob_space12 = p1[:, None] * p2[None, :]

    expect_payoff1 = np.sum(payoff_matrix_player1.reshape(dim0, -1) * prob_space23.flatten(), axis = 1)
    expect_payoff2 = np.sum(payoff_matrix_player2.transpose(1, 0, 2).reshape(dim1, -1) * prob_space13.flatten(), axis = 1)
    expect_payoff3 = np.sum(payoff_matrix_player3.transpose(2, 0, 1).reshape(dim2, -1) * prob_space12.flatten(), axis = 1)

    result = (expect_payoff1[:-1] - expect_payoff1[-1]).tolist() + \
        (expect_payoff2[:-1] - expect_payoff2[-1]).tolist() + \
        (expect_payoff3[:-1] - expect_payoff3[-1]).tolist() + \
        [sum(p1) - 1.] + [sum(p2) - 1.] + [sum(p3) - 1.]
    return result

def simulate_cs_personalized3(price, args, seed = 0):
    '''
    prices: prices for each product, list size = num_product
    sc: incremental search cost
    num_product: # of products (not including outside option)
    '''
    np.random.seed(seed)
    e0 = np.random.gumbel(0, 1, args.num_customer)
    es = np.random.gumbel(0, 1, (args.num_customer, args.num_product))
    u0 = args.a0 + args.mu * e0
    prices = np.tile(price, (args.num_customer, 1))
    a = np.tile(args.a, (args.num_customer, 1))
    u_true = a - args.ps * prices + args.mu * es
    ranks = np.argsort(u_true, axis = 1) + 1
    ranks = np.flip(ranks, 1)
    u_sort_true = np.array(list(map(lambda x, y: y[x], ranks - 1, u_true)))
    u_sort_true = np.concatenate((u0.reshape((args.num_customer, 1)), u_sort_true), axis = 1)
    u_max_sort_true = np.maximum.accumulate(u_sort_true, axis = 1)
    
    # search decisions for the first product
    ind_search = np.zeros(u_max_sort_true.shape)
    ind_search[:,:2] = 1 # always search outside option and first product

    # search decisions for the last product
    price_by_rank = np.array(list(map(lambda x, y: y[x], ranks - 1, prices)))
    a_by_rank = np.array(list(map(lambda x, y: y[x], ranks - 1, a)))
    b = (u_max_sort_true[:,:-1] + args.ps * price_by_rank - a_by_rank)/args.mu
    search_gain = args.mu * (np.euler_gamma - b[:, -1] - special.expi(- np.exp(-b[:, -1])))
    ind_search[:,-1] = (search_gain - args.sc * np.tile(1, args.num_customer)) > 0

    # search decisions for the second product
    e2_expect = np.random.gumbel(0, 1, (args.num_customer, args.n_sim)).flatten()
    a2_expect = np.repeat(a_by_rank[:, 1], args.n_sim)
    p2_expect = np.repeat(price_by_rank[:, 1], args.n_sim)
    u2_expect = a2_expect - args.ps * p2_expect +args. mu * e2_expect
    u_max_expect = np.vstack((np.repeat(u_max_sort_true[:, 1], args.n_sim), u2_expect))
    u_max_expect = np.max(u_max_expect, axis = 0)
    p3 = np.repeat(price_by_rank[:, 2], args.n_sim)
    a3 = np.repeat(a_by_rank[:, 2], args.n_sim)
    b_expect = (u_max_expect + args.ps * p3 - a3)/args.mu
    search_gain = args.mu * (np.euler_gamma - b_expect - special.expi(- np.exp(-b_expect)))
    ind_search_expect2 = (search_gain - args.sc * np.tile(1, args.num_customer * args.n_sim)) > 0
    v3 = (a3 - args.ps * p3) * (1 - np.exp(-np.exp(-b_expect))) + \
        u_max_expect * np.exp(-np.exp(-b_expect)) + \
        args.mu * (np.euler_gamma - special.expi(- np.exp(-b_expect)) - b_expect * np.exp(-np.exp(-b_expect)))
    v2 = (1 - ind_search_expect2) * u_max_expect + ind_search_expect2 * (v3 - args.sc)
    v2 = v2.reshape((args.num_customer, args.n_sim)).mean(axis = 1)
    ind_search[:, 2] = (v2 - u_max_sort_true[:, 1] - args.sc) > 0

    # purchase decisions
    search_utility = ind_search * u_sort_true
    search_utility[:,-1] = np.where(search_utility[:,-1] == 0, -np.Inf, search_utility[:,-1])
    ind_buy = np.argmax(search_utility, axis = 1)
    ind_buy_bool = ind_buy[:, None] == np.arange(1, args.num_product+1)
    
    # calculate consumer surplus
    total_utility = search_utility[np.arange(len(search_utility)), ind_buy.tolist()]
    total_search_cost = (np.sum(ind_search, axis = 1) - 2) * args.sc
    consumer_surplus = np.sum(total_utility) - np.sum(total_search_cost)
    consumer_surplus /= args.num_customer
    return consumer_surplus

def gen_unper_ranking_unper3(a, price, gamma):
    a = np.array(a)
    price = np.array(price)
    rank_score = -(a + gamma * price)
    rank = np.argsort(rank_score)
    rank += 1
    return rank.tolist()

def simulate_demand_unpersonalized3(price, rank, args, seed = 0, gen_consumer_surplus = False):
    '''
    prices: prices for each product, list size = num_product
    rank: ranking for each product, list size = num_product
        e.g., rank = [2, 3, 1] means product 2 is ranked on the first position
                                    product 3 is ranked on the second position
                                    product 1 is ranked on the third position
    sc: incremental search cost
    num_product: # of products (not including outside option)
    '''
    np.random.seed(seed)
    a0 = args.a0
    a = args.a
    mu = args.mu
    ps = args.ps
    sc = args.sc
    num_product = args.num_product
    num_customer = args.num_customer
    n_sim = args.n_sim

    e0 = np.random.gumbel(0, 1, num_customer)
    es = np.random.gumbel(0, 1, (num_customer, num_product))
    u0 = a0 + mu * e0
    prices = np.tile(price, (num_customer, 1))
    a = np.tile(a, (num_customer, 1))
    u_true = a - ps * prices + mu * es
    ranks = np.tile(rank, (num_customer, 1))
    u_sort_true = np.array(list(map(lambda x, y: y[x], ranks - 1, u_true)))
    u_sort_true = np.concatenate((u0.reshape((num_customer, 1)), u_sort_true), axis = 1)
    u_max_sort_true = np.maximum.accumulate(u_sort_true, axis = 1)
    price_by_rank = np.array(list(map(lambda x, y: y[x], ranks - 1, prices)))
    a_by_rank = np.array(list(map(lambda x, y: y[x], ranks - 1, a)))
    
    # search decisions for the outside option and top product
    ind_search = np.zeros(u_max_sort_true.shape)
    ind_search[:, :2] = 1 # always search outside option and the first product
    
    # search decisions for the last product (terminal case)
    b = (u_max_sort_true[:,:-1] + ps * price_by_rank - a_by_rank)/mu
    search_gain = mu * (np.euler_gamma - b[:, -1] - special.expi(- np.exp(-b[:, -1])))
    ind_search[:,-1] = (search_gain - sc * np.tile(1, num_customer)) > 0
    
    # search decisions for the second product
    # sample expected epsilon for the second product before searching product 2
    e2_expect = np.random.gumbel(0, 1, (num_customer, n_sim))
    e2_expect = e2_expect.flatten()
    a2_expect = np.repeat(a_by_rank[:, 1], n_sim)
    p2_expect = np.repeat(price_by_rank[:, 1], n_sim)
    u2_expect = a2_expect - ps * p2_expect + mu * e2_expect
    u_max_expect = np.vstack((np.repeat(u_max_sort_true[:, 1], n_sim), u2_expect))
    u_max_expect = np.max(u_max_expect, axis = 0)
    p3 = np.repeat(price_by_rank[:, 2], n_sim)
    a3 = np.repeat(a_by_rank[:, 2], n_sim)
    b_expect = (u_max_expect + ps * p3 - a3)/mu
    search_gain = mu * (np.euler_gamma - b_expect - special.expi(- np.exp(-b_expect)))
    ind_search_expect2 = (search_gain - sc * np.tile(1, num_customer * n_sim)) > 0
    v3 = (a3 - ps * p3) * (1 - np.exp(-np.exp(-b_expect))) + \
        u_max_expect * np.exp(-np.exp(-b_expect)) + \
        mu * (np.euler_gamma - special.expi(- np.exp(-b_expect)) - b_expect * np.exp(-np.exp(-b_expect))) 
    v2 = (1 - ind_search_expect2) * u_max_expect + ind_search_expect2 * (v3 - sc)    
    v2 = v2.reshape((num_customer, n_sim)).mean(axis = 1)
    ind_search[:, 2] = (v2 - u_max_sort_true[:, 1] - sc) > 0

    search_utility = ind_search * u_sort_true
    search_utility[:,2:] = np.where(search_utility[:,2:] == 0, -np.Inf, search_utility[:,2:])
    ind_buy = np.argmax(search_utility, axis = 1)

    demand = np.sum(ind_buy[:, None] == np.arange(1, num_product+1), axis = 0) / num_customer
    rank_index = [rank.index(i) for i in np.arange(1, num_product + 1)]
    demand = demand[rank_index]

    total_utility = search_utility[np.arange(len(search_utility)), ind_buy.tolist()]
    total_search_cost = (np.sum(ind_search, axis = 1) - 2) * sc
    consumer_surplus = np.sum(total_utility) - np.sum(total_search_cost)
    consumer_surplus /= num_customer

    if gen_consumer_surplus:
        return consumer_surplus
    else:
        return demand

def gen_profit_unpersonalized3(price, args, seed = 0):
    rank = gen_unper_ranking_unper3(args.a, price, args.gamma)
    demand = simulate_demand_unpersonalized3(price, rank, args, seed)
    profit1 = demand[0] * (price[0] - args.mc[0])
    profit2 = demand[1] * (price[1] - args.mc[1])
    profit3 = demand[2] * (price[2] - args.mc[2])
    return [profit1, profit2, profit3]

def gen_price_range_unpersonalized3(args, seed = 0):
    assert args.ranking_type == "unpersonalized3"
    sc = args.sc
    step_size = args.step_size_init
    num_product = args.num_product
    num_customer = args.num_customer
    n_sim = args.n_sim
    
    reward_file_name = f'reward_init_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
{args.num_customer}_{args.n_sim}_{args.step_size_init}.pkl'
    if os.path.exists(args.data_dir + reward_file_name):
        with open(args.data_dir + reward_file_name, 'rb') as f:
            reward_init = pickle.load(f)
        reward_matrix1 = reward_init['reward_matrix1']
        reward_matrix2 = reward_init['reward_matrix2']
        reward_matrix3 = reward_init['reward_matrix3']
        price_space1 = reward_init['price_space1']
        price_space2 = reward_init['price_space2']
        price_space3 = reward_init['price_space3']
        print(f"Loaded reward matrix from {reward_file_name}.")
    else:
        price_space1 = np.arange(args.mc[0], args.a[0] + 1.0, step_size)
        price_space1 = np.around(price_space1, 3).tolist()
        price_space2 = np.arange(args.mc[1], args.a[1] + 1.0, step_size)
        price_space2 = np.around(price_space2, 3).tolist()
        price_space3 = np.arange(args.mc[2], args.a[2] + 1.0, step_size)
        price_space3 = np.around(price_space3, 3).tolist()
        # generate combinations of prices
        price_space = [price_space1, price_space2, price_space3]
        price_space = list(list(tup) for tup in itertools.product(*price_space))
        print(f"The length of price space is {len(price_space)}.")
        reward_matrix1 = np.zeros(len(price_space))
        reward_matrix2 = np.zeros(len(price_space))
        reward_matrix3 = np.zeros(len(price_space))
        price_space = np.array(price_space)
        rank_space = gen_unper_ranking_unper3(args.a, price_space, args.gamma)
        # parallel computing the reward matrix
        print(f"There are {mp.cpu_count()} CPUs available.")
        pool = mp.Pool(mp.cpu_count())
        demand_spaces = pool.starmap(simulate_demand_unpersonalized3, [(price_space[i], rank_space[i], args, seed) for i in range(len(price_space))])
        pool.close()
        print("Finished computing reward matrix.")
        for i in range(len(price_space)):
            reward_matrix1[i] = demand_spaces[i][0] * (price_space[i][0] - args.mc[0])
            reward_matrix2[i] = demand_spaces[i][1] * (price_space[i][1] - args.mc[1])
            reward_matrix3[i] = demand_spaces[i][2] * (price_space[i][2] - args.mc[2])
        reward_matrix1 = reward_matrix1.reshape(len(price_space1), len(price_space2), len(price_space3))
        reward_matrix2 = reward_matrix2.reshape(len(price_space1), len(price_space2), len(price_space3))
        reward_matrix3 = reward_matrix3.reshape(len(price_space1), len(price_space2), len(price_space3))
        reward_matrices = {"reward_matrix1": reward_matrix1, 
                           "reward_matrix2": reward_matrix2, 
                           "reward_matrix3": reward_matrix3,
                           "price_space1": price_space1,
                           "price_space2": price_space2,
                           "price_space3": price_space3,
                           "args": args}
        with open(args.data_dir + reward_file_name, "wb") as f:
            pickle.dump(reward_matrices, f)
        print(f"Finished saving reward matrix to {reward_file_name}.")

    flag = True
    tmp_reward_matrix1 = reward_matrix1
    tmp_reward_matrix2 = reward_matrix2
    tmp_reward_matrix3 = reward_matrix3
    old_tmp_reward_matrix1 = copy.deepcopy(tmp_reward_matrix1)
    old_tmp_reward_matrix2 = copy.deepcopy(tmp_reward_matrix2)
    old_tmp_reward_matrix3 = copy.deepcopy(tmp_reward_matrix3)
    mask1_list = []
    mask2_list = []
    mask3_list = []
    while flag:
        dim0, dim1, dim2 = old_tmp_reward_matrix1.shape
        ind_dominant_reward_matrix1 = (np.max(tmp_reward_matrix1, axis = 0, keepdims = True) - tmp_reward_matrix1) > 0
        mask1 = np.sum(ind_dominant_reward_matrix1.reshape(dim0, -1), axis = 1) == (dim1 * dim2)
        dim0, dim1, dim2 = old_tmp_reward_matrix2.shape
        ind_dominant_reward_matrix2 = (np.max(tmp_reward_matrix2, axis = 1, keepdims = True) - tmp_reward_matrix2) > 0
        mask2 = np.sum(ind_dominant_reward_matrix2.transpose(1, 0, 2).reshape(dim1, -1), axis = 1) == (dim0 * dim2)
        dim0, dim1, dim2 = old_tmp_reward_matrix3.shape
        ind_dominant_reward_matrix3 = (np.max(tmp_reward_matrix3, axis = 2, keepdims = True) - tmp_reward_matrix3) > 0
        mask3 = np.sum(ind_dominant_reward_matrix3.transpose(2, 1, 0).reshape(dim2, -1), axis = 1) == (dim0 * dim1)
        tmp_reward_matrix1 = tmp_reward_matrix1[~mask1, :, :][:, ~mask2, :][:, :, ~mask3]
        tmp_reward_matrix2 = tmp_reward_matrix2[~mask1, :, :][:, ~mask2, :][:, :, ~mask3]
        tmp_reward_matrix3 = tmp_reward_matrix3[~mask1, :, :][:, ~mask2, :][:, :, ~mask3]
        mask1_list.append(mask1)
        mask2_list.append(mask2)
        mask3_list.append(mask3)
        flag_num = (old_tmp_reward_matrix1.tolist() == tmp_reward_matrix1.tolist()) + (old_tmp_reward_matrix2.tolist() == tmp_reward_matrix2.tolist()) + (old_tmp_reward_matrix3.tolist() == tmp_reward_matrix3.tolist())
        if flag_num == 3:
            flag = False
        old_tmp_reward_matrix1 = copy.deepcopy(tmp_reward_matrix1)
        old_tmp_reward_matrix2 = copy.deepcopy(tmp_reward_matrix2)
        old_tmp_reward_matrix3 = copy.deepcopy(tmp_reward_matrix3)
    
    price_space1 = np.array(price_space1)
    price_space2 = np.array(price_space2)
    price_space3 = np.array(price_space3)

    deg_price_space1 = copy.deepcopy(price_space1[~mask1_list[0]])
    for i in range(len(mask1_list) - 1):
        mask1 = mask1_list[i + 1]
        deg_price_space1 = copy.deepcopy(deg_price_space1[~mask1])
    deg_price_space2 = copy.deepcopy(price_space2[~mask2_list[0]])
    for i in range(len(mask2_list) - 1):
        mask2 = mask2_list[i + 1]
        deg_price_space2 = copy.deepcopy(deg_price_space2[~mask2])
    deg_price_space3 = copy.deepcopy(price_space3[~mask3_list[0]])
    for i in range(len(mask3_list) - 1):
        mask3 = mask3_list[i + 1]
        deg_price_space3 = copy.deepcopy(deg_price_space3[~mask3])
    
    # competitive price
    comp_prices = [min(deg_price_space1), min(deg_price_space2), min(deg_price_space3)]

    payoff_matrix_player1 = old_tmp_reward_matrix1
    payoff_matrix_player2 = old_tmp_reward_matrix2
    payoff_matrix_player3 = old_tmp_reward_matrix3
    dim0, dim1, dim2 = payoff_matrix_player1.shape
    p1_init = np.random.rand(dim0)  # [0, 1]
    p1_init = p1_init/np.sum(p1_init)
    p2_init = np.random.rand(dim1)
    p2_init = p2_init/np.sum(p2_init)
    p3_init = np.random.rand(dim2)
    p3_init = p3_init/np.sum(p3_init)
    # least squares
    p_init = p1_init.tolist() + p2_init.tolist() + p3_init.tolist()
    res = least_squares(obj_func_ls, p_init, args = (payoff_matrix_player1, payoff_matrix_player2, payoff_matrix_player3), bounds = (0, 1))
    # res = fsolve(obj_func_ls, p_init, args = (payoff_matrix_player1, payoff_matrix_player2, payoff_matrix_player3))
    p1 = res.x[:dim0]
    p2 = res.x[dim0:dim0 + dim1]
    p3 = res.x[dim0 + dim1:]
    prob_space23 = p2[:, None] * p3[None, :]
    prob_space13 = p1[:, None] * p3[None, :]
    prob_space12 = p1[:, None] * p2[None, :]
    exp_profit1 = np.sum(payoff_matrix_player1.reshape(dim0, -1) * prob_space23.flatten(), axis = 1)
    exp_profit2 = np.sum(payoff_matrix_player2.transpose(1, 0, 2).reshape(dim1, -1) * prob_space13.flatten(), axis = 1)
    exp_profit3 = np.sum(payoff_matrix_player3.transpose(2, 0, 1).reshape(dim2, -1) * prob_space12.flatten(), axis = 1)
    print(exp_profit1)
    print(exp_profit2)
    print(exp_profit3)
    comp_profit = (np.mean(exp_profit1) + np.mean(exp_profit2) + np.mean(exp_profit3))/3
    
    total_reward_matrix = (reward_matrix1 + reward_matrix2 + reward_matrix1)/3
    mono_profit = np.max(total_reward_matrix)
    mono_price_index = np.unravel_index(np.argmax(total_reward_matrix), total_reward_matrix.shape)
    mono_price1 = price_space1[mono_price_index[0]]
    mono_price2 = price_space2[mono_price_index[1]]
    mono_price3 = price_space3[mono_price_index[2]]
    mono_prices = [mono_price1, mono_price2, mono_price3]
    
    return comp_prices, mono_prices, comp_profit, mono_profit

def gen_action_reward_space_unpersonalized3(comp_prices, mono_prices, args, seed = 0):
    num_product = args.num_product

    action_spaces = []
    for i in range(num_product):
        start_price = comp_prices[i]
        end_price = mono_prices[i]
        diff = end_price - start_price
        start_price -= args.xi * diff
        end_price += args.xi * diff
        start_price = round(start_price, 3)
        end_price = round(end_price, 3)
        if args.learning_algo == "Q-learning":
            action_space = np.arange(start_price, end_price, args.step_size)
        elif "UCB" in args.learning_algo:
            action_space = np.linspace(start_price, end_price, args.n_action)
        action_space = np.around(action_space, 3).tolist()
        action_spaces.append(action_space)
    
    state_space = list(list(tup) for tup in itertools.product(*action_spaces))
    
    reward_space = []
    file_name = f"reward_space_{args.ranking_type}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}\
_{args.num_customer}_{args.n_sim}_{args.step_size_init}.pkl"
    if os.path.exists(args.data_dir + file_name):
        with open(args.data_dir + file_name, 'rb') as f:
            results = pickle.load(f)
        action_spaces = results['action_spaces']
        state_space = results['state_space']
        reward_space = results['reward_space']
        print(f"Loaded reward space from {file_name}.")
    else:
        print(f"Start generating reward space.")
        pool = mp.Pool(mp.cpu_count()-1)
        reward_space = pool.starmap(gen_profit_unpersonalized3, [(state_space[i], args) for i in range(len(state_space))])
        pool.close()
        results = {"action_spaces": action_spaces, "state_space": state_space, "reward_space": reward_space}
        with open(args.data_dir + file_name, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved reward space to {file_name}.")

def gen_mpe_sales(args, init_seed = 0):
    assert args.ranking_type == "unpersonalized2"
    assert args.score_type == "sales"
    np.random.seed(init_seed)
    action_size = args.n_action

    action_spaces = []
    for i in range(args.num_product):
        action_spaces.append(np.linspace(args.mc[i]+0.1, args.a[i], action_size).tolist())
    price_spaces = list(list(tup) for tup in itertools.product(*action_spaces))    
    demand_spaces = []
    demand_spaces1 = []
    demand_spaces0 = []
    for price in price_spaces:
        demand = gen_demand_unper_two_product(price, args.a0, args.a, args.mu, args.ps, args.sc, 1)
        demand_spaces1.append(demand)
        demand = gen_demand_unper_two_product(price, args.a0, args.a, args.mu, args.ps, args.sc, 0)
        demand_spaces0.append(demand)
    demand_spaces = demand_spaces1 + demand_spaces0
    ranking_prob_spaces = []
    for demand in demand_spaces:
        _, rank_prob = gen_sales_ranking(demand, args.kappa)
        ranking_prob_spaces.append(rank_prob.tolist())
    state_size = len(demand_spaces)
    V = np.zeros((state_size, args.num_product))
    V_new = copy.deepcopy(V)
    sigma = np.random.choice(action_size, size = (state_size, args.num_product))
    sigma_new = copy.deepcopy(sigma)
    crit = 10
    flag = False
    count = 0
    while crit > 1e-3:
        for n in range(args.num_product):
            inner_crit = 10
            while inner_crit > 1e-3:
                v = np.zeros((state_size, action_size))
                for i in range(state_size): # i is the state index
                    competitor_price = action_spaces[int(1 - n)][sigma_new[i, int(1 - n)]]
                    competitor_price = np.repeat(competitor_price, action_size)
                    focal_price = action_spaces[n]
                    if n == 0:
                        prices = np.vstack([focal_price, competitor_price]).T
                    elif n == 1:
                        prices = np.vstack([competitor_price, focal_price]).T
                    for j in range(action_size):
                        price = prices[j].tolist()
                        price_index = price_spaces.index(price)
                        reward1 = demand_spaces1[price_index][n] * (price[n] - args.mc[n])
                        reward2 = demand_spaces0[price_index][n] * (price[n] - args.mc[n])
                        rank_prob = ranking_prob_spaces[i]
                        expected_reward = rank_prob[0] * reward1 + rank_prob[1] * reward2
                        next_state1 = demand_spaces.index(demand_spaces1[price_index])
                        next_state2 = demand_spaces.index(demand_spaces0[price_index])
                        v[i, j] = expected_reward + args.delta * (rank_prob[0] * V[next_state1, n] + rank_prob[1] * V[next_state2, n])
                    V_new[i, n] = np.max(v[i, :])
                    sigma_new[i, n] = np.argmax(v[i, :])
                inner_crit = np.max(np.abs(V_new - V))
                V = copy.deepcopy(V_new)
        crit = np.max(np.abs(sigma_new - sigma))
        sigma = copy.deepcopy(sigma_new)
        print(f"Current iteration is {count} and the criterion is {crit}.")
        count += 1
        if count > 50:
            flag = True
            comp_prices = None
            return comp_prices, action_spaces, sigma, flag
    
    policy_prices = []
    for i in range(state_size):
        # print(f"Last period demand {demand_spaces[i]} has the price {action_spaces[0][sigma_new[i, 0]], action_spaces[1][sigma_new[i, 1]]}.")
        policy_prices.append([action_spaces[0][sigma_new[i, 0]], action_spaces[1][sigma_new[i, 1]]])
    comp_prices = np.min(policy_prices, axis = 0).tolist()
    print(f"Competitive prices are {comp_prices}.")
    return comp_prices, action_spaces, sigma, flag

def gen_mpe_monopoly_sales(args, init_seed = 0):
    assert args.ranking_type == "unpersonalized2"
    assert args.score_type == "sales"
    np.random.seed(init_seed)

    action_spaces = []
    for i in range(args.num_product):
        action_spaces.append(np.linspace(args.mc[i]+0.1, args.a[i], args.n_action).tolist())
    price_spaces = list(list(tup) for tup in itertools.product(*action_spaces))
    action_size = len(price_spaces)

    demand_spaces = []
    demand_spaces1 = []
    demand_spaces0 = []
    for price in price_spaces:
        demand = gen_demand_unper_two_product(price, args.a0, args.a, args.mu, args.ps, args.sc, 1)
        demand_spaces1.append(demand)
        demand = gen_demand_unper_two_product(price, args.a0, args.a, args.mu, args.ps, args.sc, 0)
        demand_spaces0.append(demand)
    demand_spaces = demand_spaces1 + demand_spaces0

    ranking_prob_spaces = []
    for demand in demand_spaces:
        _, rank_prob = gen_sales_ranking(demand, args.kappa)
        ranking_prob_spaces.append(rank_prob.tolist())

    state_size = len(demand_spaces)
    V = np.zeros(state_size)
    V_new = copy.deepcopy(V)
    sigma = np.random.choice(action_size, size = state_size)
    sigma_new = copy.deepcopy(sigma)

    crit = 10
    flag = False
    count = 0
    while crit > 1e-3:
        inner_crit = 10
        while inner_crit > 1e-3:
            v = np.zeros((state_size, action_size))
            for i in range(state_size):
                state = demand_spaces[i]
                for j in range(action_size):
                    price = price_spaces[j]
                    price_index = price_spaces.index(price)
                    rank_prob = ranking_prob_spaces[i]
                    reward1 = demand_spaces1[price_index][0] * (price[0] - args.mc[0]) + demand_spaces1[price_index][1] * (price[1] - args.mc[1])
                    reward2 = demand_spaces0[price_index][0] * (price[0] - args.mc[0]) + demand_spaces0[price_index][1] * (price[1] - args.mc[1])
                    expect_reward = rank_prob[0] * reward1 + rank_prob[1] * reward2
                    next_state1 = demand_spaces.index(demand_spaces1[price_index])
                    next_state2 = demand_spaces.index(demand_spaces0[price_index])
                    v[i, j] = expect_reward + args.delta * (rank_prob[0] * V[next_state1] + rank_prob[1] * V[next_state2])
                V_new[i] = np.max(v[i, :])
                sigma_new[i] = np.argmax(v[i, :])
            inner_crit = np.max(np.abs(V_new - V))
            V = copy.deepcopy(V_new)
        crit = np.max(np.abs(sigma_new - sigma))
        sigma = copy.deepcopy(sigma_new)
        print(f"Current iteration is {count} and the criterion is {crit}.")

        count += 1
        if count > 50:
            flag = True
            mon_prices = None
            return mon_prices, action_spaces, sigma, flag
    
    policy_prices = []
    for i in range(state_size):
        # print(f"Last period demand {demand_spaces[i]} has the price {price_spaces[sigma_new[i]]}.")
        policy_prices.append(price_spaces[sigma_new[i]])
    mon_prices = np.max(policy_prices, axis = 0).tolist()
    print(f"Monopoly prices are {mon_prices}.")

    return mon_prices, action_spaces, sigma, flag

def gen_price_range_mpe(args, seed = 0):
    assert args.ranking_type == "unpersonalized2"
    assert args.score_type == "sales"

    file_name = f"mpe_{args.ranking_type}_{args.file_prefix}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}\
_{args.step_size_init}_nactions{args.n_action}.pkl"
    if os.path.exists(args.data_dir + file_name):
        with open(args.data_dir + file_name, "rb") as f:
            results = pickle.load(f)
        comp_prices_list = results["comp_prices_list"]
        mono_prices_list = results["mono_prices_list"]
        print(f"Loaded MPE prices from {file_name}.")
        comp_prices = np.min(comp_prices_list, axis = 0)
        mono_prices = np.max(mono_prices_list, axis = 0)
    else:
        np.random.seed(seed)
        total_mpe = 0
        total_monopoly = 0
        seeds_mpe = []
        seeds_monopoly = []
        comp_prices_list = []
        mono_prices_list = []
        comp_action_spaces_list = []
        comp_sigma_list = []
        mono_action_spaces_list = []
        mono_sigma_list = []
        while total_mpe < args.n_mpe:
            init_seed = np.random.randint(0, 100000)
            comp_prices, action_spaces, sigma, flag = gen_mpe_sales(args, init_seed)
            if flag:
                print("Failed to find MPE.")
            else:
                print(f"Seed is {init_seed}.")
                print(f"Competitive prices are {comp_prices}.")
                total_mpe += 1
                seeds_mpe.append(init_seed)
                comp_prices_list.append(comp_prices)
                comp_action_spaces_list.append(action_spaces)
                comp_sigma_list.append(sigma)
        while total_monopoly < args.n_mpe:
            init_seed = np.random.randint(0, 100000)
            mono_prices, action_spaces, sigma, flag = gen_mpe_monopoly_sales(args, init_seed)
            if flag:
                print("Failed to find monopoly prices.")
            else:
                print(f"Seed is {init_seed}.")
                print(f"Monopoly prices are {mono_prices}.")
                total_monopoly += 1
                seeds_monopoly.append(init_seed)
                mono_prices_list.append(mono_prices)
                mono_action_spaces_list.append(action_spaces)
                mono_sigma_list.append(sigma)
        results = {"seeds_mpe": seeds_mpe, 
                    "seeds_monopoly": seeds_monopoly, 
                    "comp_prices_list": comp_prices_list, 
                    "mono_prices_list": mono_prices_list,
                    "comp_action_spaces_list": comp_action_spaces_list,
                    "comp_sigma_list": comp_sigma_list,
                    "mono_sigma_list": mono_sigma_list,
                    "mono_action_spaces_list": mono_action_spaces_list,
                    "args": args}
        with open(args.data_dir + file_name, "wb") as f:
            pickle.dump(results, f)
        comp_prices = np.min(comp_prices_list, axis = 0)
        mono_prices = np.max(mono_prices_list, axis = 0)
    print(f"Competitive prices are {comp_prices}.")
    print(f"Monopoly prices are {mono_prices}.")
    return comp_prices, mono_prices

def simulate_demand_unpersonalized2_risk_utility(price, args, seed = 0, ranks = None, gen_consumer_surplus = False):
    '''
    meaning of omegas
    omegas = [0, 0]: no prior information for both products
    omegas = [omega, 0]: perfect prior information for product 1 with certainty level omega
    omegas = [0, omega]: perfect prior information for product 2 with certainty level omega
    '''
    np.random.seed(seed)
    a0 = args.a0
    a = args.a
    mu = args.mu
    ps = args.ps
    sc = args.sc
    num_product = args.num_product
    num_customer = args.num_customer
    n_sim = args.n_sim

    # true utility
    e0 = np.random.gumbel(0, 1, num_customer)
    A0 = a0 + mu * e0
    e1 = np.random.gumbel(0, 1, num_customer)
    e2 = np.random.gumbel(0, 1, num_customer)
    u0 = A0 - args.r * A0**2
    A1 = a[0] + mu * e1
    A2 = a[1] + mu * e2
    u1 = A1 - ps * price[0] - args.r * A1**2
    u2 = A2 - ps * price[1] - args.r * A2**2
    u_true = np.column_stack((u0, u1, u2))

    # expected utility
    A_expect = np.array(a) + mu * np.euler_gamma
    # print(f"A_expect is {A_expect}")
    var_A_expect = math.pi**2 / 6
    # print(f"var_A_expect is {var_A_expect}")
    u_expect = A_expect - ps * np.array(price) - args.r * A_expect**2 - args.r * var_A_expect
    # print(f"u_expect is {u_expect}")
    # rank the products based on expected utility
    if not ranks:
        ranks = np.argsort(u_expect, axis = 0)
        ranks = np.flip(ranks, 0)
        ranks = np.where(u_expect[0] == u_expect[1], np.array([0, 1]), ranks)
    if ranks[0] == 0:
        u_sort_true = np.column_stack((u0, u1, u2))
        e_sim = np.random.gumbel(0, 1, (num_customer, n_sim))
        A_sim = a[1] + mu * e_sim
        # print(f"A_sim is:\n {A_sim}")
        u_sim = A_sim - ps * price[1] - args.r * A_sim**2
        # print(f"u_sim is:\n {u_sim}")
    elif ranks[0] == 1:
        u_sort_true = np.column_stack((u0, u2, u1))
        e_sim = np.random.gumbel(0, 1, (num_customer, n_sim))
        A_sim = a[0] + mu * e_sim
        # print(f"A_sim is:\n {A_sim}")
        u_sim = A_sim - ps * price[0] - args.r * A_sim**2
        # print(f"u_sim is:\n {u_sim}")
    # print(f"u_sort_true is:\n {u_sort_true}")
    u_max_sort_true = np.maximum.accumulate(u_sort_true, axis = 1)
    # print(f"u_max_sort_true is:\n {u_max_sort_true}")

    # search decisions for the outside option and top product
    ind_search = np.zeros(u_true.shape)
    ind_search[:, :2] = 1 # always search outside option and the first product
    # search decisions for the second-ranked product
    search_gain = u_sim - np.repeat(u_max_sort_true[:, 1], n_sim).reshape(num_customer, n_sim)
    search_gain = np.where(search_gain < 0, 0, search_gain)
    search_gain = np.mean(search_gain, axis = 1)
    # print(f"search_gain is:\n {search_gain}")
    ind_search[:, 2] = search_gain > args.sc
    # print(f"ind_search is:\n {ind_search}")

    # purchase decisions
    search_utility = ind_search * u_sort_true
    search_utility[:,-1] = np.where(search_utility[:,-1] == 0, -np.Inf, search_utility[:,-1])
    # print(f"search_utility is:\n {search_utility}")
    ind_buy = np.argmax(search_utility, axis = 1)
    # print(f"ind_buy is:\n {ind_buy}")
    # calculate demand
    demand = np.sum(ind_buy[:, None] == np.arange(1, num_product+1), axis = 0) / num_customer
    rank_index = [list(ranks).index(i) for i in np.arange(num_product)]
    demand = demand[rank_index]
    # print(f"demand is {demand}")
    profit = [demand[0] * (price[0] - args.mc[0]), demand[1] * (price[1] - args.mc[1])]
    
    if gen_consumer_surplus:
        total_utility = search_utility[np.arange(len(search_utility)), ind_buy.tolist()]
        total_search_cost = (np.sum(ind_search, axis = 1) - 2) * sc
        consumer_surplus = np.sum(total_utility) - np.sum(total_search_cost)
        consumer_surplus /= num_customer
        return consumer_surplus
    else:
        return demand

def gen_profit_unpersonalized2_risk_utility(price, args, ranks = None):
    demand = simulate_demand_unpersonalized2_risk_utility(price, args, ranks = ranks)
    profit = [demand[0] * (price[0] - args.mc[0]), demand[1] * (price[1] - args.mc[1])]
    return profit

def simulate_cs_rational_belief_personalized2_risk_utility(price, args, seed = 0):
    np.random.seed(seed)
    a0 = args.a0
    a = args.a
    mu = args.mu
    ps = args.ps
    sc = args.sc
    num_product = args.num_product
    num_customer = args.num_customer
    n_sim = args.n_sim

    # true utility
    e0 = np.random.gumbel(0, 1, num_customer)
    A0 = a0 + mu * e0
    e1 = np.random.gumbel(0, 1, num_customer)
    e2 = np.random.gumbel(0, 1, num_customer)
    u0 = A0 - args.r * A0**2
    A1 = a[0] + mu * e1
    A2 = a[1] + mu * e2
    u1 = A1 - ps * price[0] - args.r * A1**2
    u2 = A2 - ps * price[1] - args.r * A2**2
    u_true = np.column_stack((u0, u1, u2))
    u_max_sort_true = np.maximum.accumulate(u_true, axis = 1)
    cs = np.mean(u_max_sort_true)
    return cs

def simulate_demand_personalized2_risk_utility(price, args, seed = 0, gen_consumer_surplus = False):
    np.random.seed(seed)
    a0 = args.a0
    a = args.a
    mu = args.mu
    ps = args.ps
    sc = args.sc
    num_product = args.num_product
    num_customer = args.num_customer
    n_sim = args.n_sim

    # true utility
    e0 = np.random.gumbel(0, 1, num_customer)
    A0 = a0 + mu * e0
    e1 = np.random.gumbel(0, 1, num_customer)
    e2 = np.random.gumbel(0, 1, num_customer)
    u0 = A0 - args.r * A0**2
    A1 = a[0] + mu * e1
    A2 = a[1] + mu * e2
    u1 = A1 - ps * price[0] - args.r * A1**2
    u2 = A2 - ps * price[1] - args.r * A2**2
    u_true = np.column_stack((u0, u1, u2))
    # print(f"u_true is:\n {u_true}")

    # rank the products based on actual utility
    ranks = np.argsort(np.column_stack((u1, u2)), axis = 1)
    ranks = np.flip(ranks, 1)
    # print(f"ranks is:\n {ranks}")
    prices = np.tile(price, (num_customer, 1))
    a = np.tile(a, (num_customer, 1))
    u_sort_true = np.array(list(map(lambda x, y: y[x], ranks, u_true[:, 1:])))
    # print(f"u_sort_true is:\n {u_sort_true}")
    u_sort_true = np.concatenate((u0.reshape((num_customer, 1)), u_sort_true), axis = 1)
    # print(f"u_sort_true is:\n {u_sort_true}")
    e_sim = np.random.gumbel(0, 1, (num_customer, n_sim))
    a_by_rank = np.array(list(map(lambda x, y: y[x], ranks, a)))
    # print(f"a_by_rank is:\n {a_by_rank}")
    price_by_rank = np.array(list(map(lambda x, y: y[x], ranks, prices)))
    # print(f"price_by_rank is:\n {price_by_rank}")
    A_sim = np.repeat(a_by_rank[:, 1], n_sim).reshape(num_customer, n_sim) + mu * e_sim
    u_sim = A_sim - ps * np.repeat(price_by_rank[:, 1], n_sim).reshape(num_customer, n_sim) - args.r * A_sim**2
    u_max_sort_true = np.maximum.accumulate(u_sort_true, axis = 1)
    # print(f"u_max_sort_true is:\n {u_max_sort_true}")

    # search decisions for the outside option and top product
    ind_search = np.zeros(u_true.shape)
    ind_search[:, :2] = 1 # always search outside option and the first product
    # search decisions for the second-ranked product
    search_gain = u_sim - np.repeat(u_max_sort_true[:, 1], n_sim).reshape(num_customer, n_sim)
    search_gain = np.where(search_gain < 0, 0, search_gain)
    search_gain = np.mean(search_gain, axis = 1)
    ind_search[:, 2] = search_gain > args.sc

    # purchase decisions
    search_utility = ind_search * u_sort_true
    search_utility[:,-1] = np.where(search_utility[:,-1] == 0, -np.Inf, search_utility[:,-1])
    # print(f"search_utility is:\n {search_utility}")
    ind_buy = np.argmax(search_utility, axis = 1)
    # print(f"ind_buy is:\n {ind_buy}")
    ind_buy_bool = ind_buy[:, None] == np.arange(1, args.num_product+1)
    # print(f"ind_buy_bool is:\n {ind_buy_bool}")
    # calculate demand
    ranks = ranks + 1
    # print(ranks)
    # print(ind_buy_bool * ranks)
    tmp_demand = np.sum(ind_buy_bool * ranks, axis = 1)
    # print(f"tmp_demand is {tmp_demand}.")
    product_id, counts = np.unique(tmp_demand, return_counts=True)
    demand = np.zeros(args.num_product)
    for i in range(len(product_id)):
        demand[product_id[i]-1] = counts[i]
    demand /= args.num_customer
    # print(f"demand is {demand}.")
    
    profit = [demand[0] * (price[0] - args.mc[0]), demand[1] * (price[1] - args.mc[1])]
    # print(f"Simulated demand is {demand}.")
    # print(f"Simulated profit is {profit}.")
    if gen_consumer_surplus:
        total_utility = search_utility[np.arange(len(search_utility)), ind_buy.tolist()]
        total_search_cost = (np.sum(ind_search, axis = 1) - 2) * args.sc
        consumer_surplus = np.sum(total_utility) - np.sum(total_search_cost)
        consumer_surplus /= args.num_customer
        return consumer_surplus
    else:
        return demand

def gen_price_range_unpersonalized2_risk_utility(args, seed = 0):
    price_space1 = np.arange(args.mc[0], args.a[0] + 0.5, args.step_size_init)
    price_space1 = np.around(price_space1, 3)
    m1 = len(price_space1)
    price_space2 = np.arange(args.mc[1], args.a[1] + 0.5, args.step_size_init)
    price_space2 = np.around(price_space2, 3)
    m2 = len(price_space2)
    revenue_matrix1 = np.zeros((m1, m2))
    revenue_matrix2 = np.zeros((m1, m2))
    mon_revenue_matrix1 = np.zeros((m1, m2))
    mon_revenue_matrix2 = np.zeros((m1, m2))
    
    for i in range(m1):
        p1 = price_space1[i]
        for j in range(m2):
            p2 = price_space2[j]
            profit = gen_profit_unpersonalized2_risk_utility([p1, p2], args)
            revenue_matrix1[i, j] = profit[0]
            revenue_matrix2[i, j] = profit[1]
            profit1 = gen_profit_unpersonalized2_risk_utility([p1, p2], args, ranks = [0, 1])
            profit2 = gen_profit_unpersonalized2_risk_utility([p1, p2], args, ranks = [1, 0])
            mon_revenue_matrix1[i, j] = profit1[0]
            mon_revenue_matrix2[i, j] = profit2[1]
    
    flag = True
    tmp_revenue_matrix1 = revenue_matrix1
    tmp_revenue_matrix2 = revenue_matrix2
    old_tmp_revenue_matrix1 = copy.deepcopy(tmp_revenue_matrix1)
    old_tmp_revenue_matrix2 = copy.deepcopy(tmp_revenue_matrix2)
    mask1_list = []
    mask2_list = []
    while flag:
        ind_dominant_revenue_matrix1 = (np.max(tmp_revenue_matrix1, axis = 0, keepdims = True) - tmp_revenue_matrix1) > 0
        mask1 = np.sum(ind_dominant_revenue_matrix1, axis = 1) == tmp_revenue_matrix1.shape[1]
        ind_dominant_revenue_matrix2 = (np.max(tmp_revenue_matrix2, axis = 1, keepdims = True) - tmp_revenue_matrix2) > 0
        mask2 = np.sum(ind_dominant_revenue_matrix2, axis = 0) == tmp_revenue_matrix2.shape[0]
        tmp_revenue_matrix1 = tmp_revenue_matrix1[~mask1, :][:, ~mask2]
        tmp_revenue_matrix2 = tmp_revenue_matrix2[:, ~mask2][~mask1, :]
        mask1_list.append(mask1)
        mask2_list.append(mask2)
        flag_num = (old_tmp_revenue_matrix1.tolist() == tmp_revenue_matrix1.tolist()) + (old_tmp_revenue_matrix2.tolist() == tmp_revenue_matrix2.tolist())
        if flag_num == 2:
            flag = False
        old_tmp_revenue_matrix1 = copy.deepcopy(tmp_revenue_matrix1)
        old_tmp_revenue_matrix2 = copy.deepcopy(tmp_revenue_matrix2)
    
    # calculate mixed strategy for player 1
    m2 = len(tmp_revenue_matrix2)
    p1_init = np.array([1/m2] * m2)
    bounds1 = [[0] * m2, [1] * m2]
    p1 = least_squares(obj_func, p1_init, bounds = bounds1, args = (tmp_revenue_matrix2,))
    p1 = p1.x
    deg_price_space1 = copy.deepcopy(price_space1[~mask1_list[0]])
    for i in range(len(mask1_list) - 1):
        mask1 = mask1_list[i + 1]
        deg_price_space1 = copy.deepcopy(deg_price_space1[~mask1])
    # calculate mixed strategy for player 2
    m1 = len(tmp_revenue_matrix1.T)
    p2_init = np.array([1/m1] * m1)
    bounds2 = [[0] * m1, [1] * m1]
    p2 = least_squares(obj_func, p2_init, bounds = bounds2, args = (tmp_revenue_matrix1.T,))
    p2 = p2.x
    deg_price_space2 = copy.deepcopy(price_space2[~mask2_list[0]])
    for i in range(len(mask2_list) - 1):
        mask2 = mask2_list[i + 1]
        deg_price_space2 = copy.deepcopy(deg_price_space2[~mask2])
    comp_prices = [min(deg_price_space1), min(deg_price_space2)]
    # calculate monopoly price for player 1
    mon1 = max(price_space1[np.argmax(mon_revenue_matrix1, axis = 0)])
    mon2 = max(price_space2[np.argmax(mon_revenue_matrix2, axis = 1)])
    mono_prices = [mon1, mon2]
    return comp_prices, mono_prices

def gen_profit_personalized2_risk_utility(price, args):
    demand = simulate_demand_personalized2_risk_utility(price, args)
    profit = [demand[0] * (price[0] - args.mc[0]), demand[1] * (price[1] - args.mc[1])]
    return profit

def gen_reward_space_personalized2_risk_utility(comp_prices, mono_prices, args):
    action_spaces = []
    for i in range(args.num_product):
        start_price = comp_prices[i]
        end_price = mono_prices[i]
        diff = end_price - start_price
        start_price -= args.xi * diff
        end_price += args.xi * diff
        start_price = round(start_price, 3)
        end_price = round(end_price, 3)
        if args.learning_algo == "Q-learning":
            action_space = np.arange(start_price, end_price, args.step_size)
        elif "UCB" in args.learning_algo:
            action_space = np.linspace(start_price, end_price, args.n_action)
        action_space = np.around(action_space, 3).tolist()
        action_spaces.append(action_space)
    state_space = list(list(tup) for tup in itertools.product(*action_spaces))

    reward_space = []
    for i in range(len(state_space)):
        rewards = gen_profit_personalized2_risk_utility(state_space[i], args)
        reward_space.append(np.around(rewards, 3).tolist())
    
    file_name = f"reward_space_{args.ranking_type}_{args.file_prefix}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}\
_{args.num_customer}_{args.n_sim}_{args.r}.pkl"
    results = {"state_space": state_space, "reward_space": reward_space}
    with open(args.data_dir + file_name, 'wb') as f:
        pickle.dump(results, f)
    
def gen_reward_space_unpersonazlied2_risk_utility(comp_prices, mono_prices, args):
    action_spaces = []
    for i in range(args.num_product):
        start_price = comp_prices[i]
        end_price = mono_prices[i]
        diff = end_price - start_price
        start_price -= args.xi * diff
        end_price += args.xi * diff
        start_price = round(start_price, 3)
        end_price = round(end_price, 3)
        if args.learning_algo == "Q-learning":
            action_space = np.arange(start_price, end_price, args.step_size)
        elif args.learning_algo == "UCB":
            action_space = np.linspace(start_price, end_price, args.n_action)
        action_space = np.around(action_space, 3).tolist()
        action_spaces.append(action_space)
    state_space = list(list(tup) for tup in itertools.product(*action_spaces))

    reward_space = []
    for i in range(len(state_space)):
        rewards = gen_profit_unpersonalized2_risk_utility(state_space[i], args)
        reward_space.append(np.around(rewards, 3).tolist())
    
    file_name = f"reward_space_{args.ranking_type}_{args.file_prefix}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}\
_{args.num_customer}_{args.n_sim}_{args.r}.pkl"
    results = {"state_space": state_space, "reward_space": reward_space}
    with open(args.data_dir + file_name, 'wb') as f:
        pickle.dump(results, f)

def gen_price_range_personalized2_risk_utility(args, seed = 0):
    price_space1 = np.arange(args.mc[0], args.a[0] + 0.5, args.step_size_init)
    price_space1 = np.around(price_space1, 3)
    m1 = len(price_space1)
    price_space2 = np.arange(args.mc[1], args.a[1] + 0.5, args.step_size_init)
    price_space2 = np.around(price_space2, 3)
    m2 = len(price_space2)
    revenue_matrix1 = np.zeros((m1, m2))
    revenue_matrix2 = np.zeros((m1, m2))
    
    for i in range(m1):
        p1 = price_space1[i]
        for j in range(m2):
            p2 = price_space2[j]
            profit = gen_profit_personalized2_risk_utility([p1, p2], args)
            revenue_matrix1[i, j] = profit[0]
            revenue_matrix2[i, j] = profit[1]  

    flag = True
    tmp_revenue_matrix1 = revenue_matrix1
    tmp_revenue_matrix2 = revenue_matrix2
    old_tmp_revenue_matrix1 = copy.deepcopy(tmp_revenue_matrix1)
    old_tmp_revenue_matrix2 = copy.deepcopy(tmp_revenue_matrix2)
    mask1_list = []
    mask2_list = []
    while flag:
        ind_dominant_revenue_matrix1 = (np.max(tmp_revenue_matrix1, axis = 0, keepdims = True) - tmp_revenue_matrix1) > 0
        mask1 = np.sum(ind_dominant_revenue_matrix1, axis = 1) == tmp_revenue_matrix1.shape[1]
        ind_dominant_revenue_matrix2 = (np.max(tmp_revenue_matrix2, axis = 1, keepdims = True) - tmp_revenue_matrix2) > 0
        mask2 = np.sum(ind_dominant_revenue_matrix2, axis = 0) == tmp_revenue_matrix2.shape[0]
        tmp_revenue_matrix1 = tmp_revenue_matrix1[~mask1, :][:, ~mask2]
        tmp_revenue_matrix2 = tmp_revenue_matrix2[:, ~mask2][~mask1, :]
        mask1_list.append(mask1)
        mask2_list.append(mask2)
        flag_num = (old_tmp_revenue_matrix1.tolist() == tmp_revenue_matrix1.tolist()) + (old_tmp_revenue_matrix2.tolist() == tmp_revenue_matrix2.tolist())
        if flag_num == 2:
            flag = False
        old_tmp_revenue_matrix1 = copy.deepcopy(tmp_revenue_matrix1)
        old_tmp_revenue_matrix2 = copy.deepcopy(tmp_revenue_matrix2)
    
    # calculate mixed strategy for player 1
    m2 = len(tmp_revenue_matrix2)
    p1_init = np.array([1/m2] * m2)
    bounds1 = [[0] * m2, [1] * m2]
    p1 = least_squares(obj_func, p1_init, bounds = bounds1, args = (tmp_revenue_matrix2,))
    p1 = p1.x
    deg_price_space1 = copy.deepcopy(price_space1[~mask1_list[0]])
    for i in range(len(mask1_list) - 1):
        mask1 = mask1_list[i + 1]
        deg_price_space1 = copy.deepcopy(deg_price_space1[~mask1])
    # calculate mixed strategy for player 2
    m1 = len(tmp_revenue_matrix1.T)
    p2_init = np.array([1/m1] * m1)
    bounds2 = [[0] * m1, [1] * m1]
    p2 = least_squares(obj_func, p2_init, bounds = bounds2, args = (tmp_revenue_matrix1.T,))
    p2 = p2.x
    deg_price_space2 = copy.deepcopy(price_space2[~mask2_list[0]])
    for i in range(len(mask2_list) - 1):
        mask2 = mask2_list[i + 1]
        deg_price_space2 = copy.deepcopy(deg_price_space2[~mask2])
    comp_prices = [min(deg_price_space1), min(deg_price_space2)]

    reward_array = revenue_matrix1 + revenue_matrix2
    ind1, ind2 = np.unravel_index(np.argmax(reward_array), np.array(reward_array).shape)
    mono_prices = [price_space1[ind1], price_space2[ind2]]    

    return comp_prices, mono_prices

@jit_integrand_function
def integrate_f_belief_update(args):
    x, diff, lamb = args
    gumbel_pdf = np.exp(-(x + np.exp(-x)))
    F_diff = np.exp(-np.exp(-diff))
    f_trunc = gumbel_pdf/(1 - F_diff)
    num = (1 - lamb) * f_trunc
    F_diff = np.exp(-np.exp(-diff))
    den = lamb * F_diff + (1 - lamb) * (1 - F_diff)
    f_update = num/den
    return f_update * (x - diff)

def simulate_imperfect_per_rational(price, 
                                    args, 
                                    seed = 0, 
                                    gen_consumer_surplus = False, 
                                    gen_ranks = False,
                                    decompose = False):
    np.random.seed(seed)
    a0 = args.a0
    a = args.a
    mu = args.mu
    ps = args.ps
    sc = args.sc
    num_product = args.num_product
    num_customer = args.num_customer
    n_sim = args.n_sim
    lamb = args.lamb

    # true utility
    e0 = np.random.gumbel(0, 1, num_customer)
    u0 = a0 + mu * e0
    e1 = np.random.gumbel(0, 1, num_customer)
    e2 = np.random.gumbel(0, 1, num_customer)
    u1 = a[0] - ps * price[0] + mu * e1
    u2 = a[1] - ps * price[1] + mu * e2
    u_true = np.column_stack((u0, u1, u2))

    ranks = np.argsort(np.column_stack((u1, u2)), axis = 1)
    ranks = np.flip(ranks, 1)
    ind_correct = np.random.uniform(0, 1, num_customer) < lamb 
    ranks = np.where(ind_correct[:, None], ranks, np.flip(ranks, 1))

    # search decisions for the outside option and top product
    ind_search = np.zeros(u_true.shape)
    ind_search[:, :2] = 1 # always search outside option and the first product
    prices = np.tile(price, (num_customer, 1))
    a = np.tile(a, (num_customer, 1))
    u_sort_true = np.array(list(map(lambda x, y: y[x], ranks, u_true[:, 1:])))
    u_sort_true = np.concatenate((u0.reshape((num_customer, 1)), u_sort_true), axis = 1)
    a_by_rank = np.array(list(map(lambda x, y: y[x], ranks, a)))
    price_by_rank = np.array(list(map(lambda x, y: y[x], ranks, prices)))
    u_max_sort_true = np.maximum.accumulate(u_sort_true, axis = 1)
    diffs = u_max_sort_true[:, 1] - a_by_rank[:, 1] + ps * price_by_rank[:, 1]
    search_gain = [integrate.quad(integrate_f_belief_update, diff, np.Inf, args = (diff, lamb))[0] for diff in diffs]
    ind_search[:, 2] = np.array(search_gain) > sc

    # purchase decisions
    search_utility = ind_search * u_sort_true
    search_utility[:,-1] = np.where(search_utility[:,-1] == 0, -np.Inf, search_utility[:,-1])
    ind_buy = np.argmax(search_utility, axis = 1)
    ind_buy_bool = ind_buy[:, None] == np.arange(1, args.num_product+1)
    ranks = ranks + 1
    tmp_demand = np.sum(ind_buy_bool * ranks, axis = 1)
    product_id, counts = np.unique(tmp_demand, return_counts=True)
    demand = np.zeros(args.num_product)
    for i in range(len(product_id)):
        demand[product_id[i]-1] = counts[i]
    demand /= args.num_customer

    if decompose:
        mask_rank12 = ranks[:, 0] == 1
        tmp_demand_rank12 = tmp_demand * mask_rank12
        product_id, counts = np.unique(tmp_demand_rank12, return_counts=True)
        demand_rank12 = np.zeros(args.num_product)
        for i in range(len(product_id)):
            demand_rank12[product_id[i]-1] = counts[i]
        demand_rank12 /= np.sum(mask_rank12)

        mask_rank21 = ranks[:, 0] == 2
        tmp_demand_rank21 = tmp_demand * mask_rank21
        product_id, counts = np.unique(tmp_demand_rank21, return_counts=True)
        demand_rank21 = np.zeros(args.num_product)
        for i in range(len(product_id)):
            demand_rank21[product_id[i]-1] = counts[i]
        demand_rank21 /= np.sum(mask_rank21)

        return demand, demand_rank12, demand_rank21, ranks

    if gen_consumer_surplus:
        total_utility = search_utility[np.arange(len(search_utility)), ind_buy.tolist()]
        total_search_cost = (np.sum(ind_search, axis = 1) - 2) * args.sc
        consumer_surplus = np.sum(total_utility) - np.sum(total_search_cost)
        consumer_surplus /= args.num_customer
        return consumer_surplus
    else:
        if gen_ranks:
            return demand, ranks
        else:
            return demand

def gen_profit_imperfect_per_rational(price, args):
    demand = simulate_imperfect_per_rational(price, args)
    profit = [demand[0] * (price[0] - args.mc[0]), demand[1] * (price[1] - args.mc[1])]
    return profit

def gen_reward_space_imperfect_per_rational(comp_prices, mono_prices, args):
    action_spaces = []
    for i in range(args.num_product):
        start_price = comp_prices[i]
        end_price = mono_prices[i]
        diff = end_price - start_price
        start_price -= args.xi * diff
        end_price += args.xi * diff
        start_price = round(start_price, 3)
        end_price = round(end_price, 3)
        if args.learning_algo == "Q-learning":
            action_space = np.arange(start_price, end_price, args.step_size)
        elif args.learning_algo == "UCB_tuned":
            action_space = np.linspace(start_price, end_price, args.n_action)
        action_space = np.around(action_space, 3).tolist()
        action_spaces.append(action_space)
    state_space = list(list(tup) for tup in itertools.product(*action_spaces))

    reward_space = []
    for i in range(len(state_space)):
        rewards = gen_profit_imperfect_per_rational(state_space[i], args)
        reward_space.append(np.around(rewards, 3).tolist())
    
    file_name = f"reward_space_{args.ranking_type}_{args.file_prefix}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}\
_{args.num_customer}_{args.n_sim}_{args.r}.pkl"
    results = {"state_space": state_space, "reward_space": reward_space}
    with open(args.data_dir + file_name, 'wb') as f:
        pickle.dump(results, f)

def gen_price_range_imperfect_personalized(args):
    price_space1 = np.arange(args.mc[0], args.a[0] + 0.5, args.step_size_init)
    price_space1 = np.around(price_space1, 3)
    m1 = len(price_space1)
    price_space2 = np.arange(args.mc[1], args.a[1] + 0.5, args.step_size_init)
    price_space2 = np.around(price_space2, 3)
    m2 = len(price_space2)
    revenue_matrix1 = np.zeros((m1, m2))
    revenue_matrix2 = np.zeros((m1, m2))
    
    for i in range(m1):
        p1 = price_space1[i]
        for j in range(m2):
            p2 = price_space2[j]
            profit = gen_profit_imperfect_per_rational([p1, p2], args)
            revenue_matrix1[i, j] = profit[0]
            revenue_matrix2[i, j] = profit[1]

    flag = True
    tmp_revenue_matrix1 = revenue_matrix1
    tmp_revenue_matrix2 = revenue_matrix2
    old_tmp_revenue_matrix1 = copy.deepcopy(tmp_revenue_matrix1)
    old_tmp_revenue_matrix2 = copy.deepcopy(tmp_revenue_matrix2)
    mask1_list = []
    mask2_list = []
    while flag:
        ind_dominant_revenue_matrix1 = (np.max(tmp_revenue_matrix1, axis = 0, keepdims = True) - tmp_revenue_matrix1) > 0
        mask1 = np.sum(ind_dominant_revenue_matrix1, axis = 1) == tmp_revenue_matrix1.shape[1]
        ind_dominant_revenue_matrix2 = (np.max(tmp_revenue_matrix2, axis = 1, keepdims = True) - tmp_revenue_matrix2) > 0
        mask2 = np.sum(ind_dominant_revenue_matrix2, axis = 0) == tmp_revenue_matrix2.shape[0]
        tmp_revenue_matrix1 = tmp_revenue_matrix1[~mask1, :][:, ~mask2]
        tmp_revenue_matrix2 = tmp_revenue_matrix2[:, ~mask2][~mask1, :]
        mask1_list.append(mask1)
        mask2_list.append(mask2)
        flag_num = (old_tmp_revenue_matrix1.tolist() == tmp_revenue_matrix1.tolist()) + (old_tmp_revenue_matrix2.tolist() == tmp_revenue_matrix2.tolist())
        if flag_num == 2:
            flag = False
        old_tmp_revenue_matrix1 = copy.deepcopy(tmp_revenue_matrix1)
        old_tmp_revenue_matrix2 = copy.deepcopy(tmp_revenue_matrix2)
    
    # calculate mixed strategy for player 1
    m2 = len(tmp_revenue_matrix2)
    p1_init = np.array([1/m2] * m2)
    bounds1 = [[0] * m2, [1] * m2]
    p1 = least_squares(obj_func, p1_init, bounds = bounds1, args = (tmp_revenue_matrix2,))
    p1 = p1.x
    deg_price_space1 = copy.deepcopy(price_space1[~mask1_list[0]])
    for i in range(len(mask1_list) - 1):
        mask1 = mask1_list[i + 1]
        deg_price_space1 = copy.deepcopy(deg_price_space1[~mask1])
    # calculate mixed strategy for player 2
    m1 = len(tmp_revenue_matrix1.T)
    p2_init = np.array([1/m1] * m1)
    bounds2 = [[0] * m1, [1] * m1]
    p2 = least_squares(obj_func, p2_init, bounds = bounds2, args = (tmp_revenue_matrix1.T,))
    p2 = p2.x
    deg_price_space2 = copy.deepcopy(price_space2[~mask2_list[0]])
    for i in range(len(mask2_list) - 1):
        mask2 = mask2_list[i + 1]
        deg_price_space2 = copy.deepcopy(deg_price_space2[~mask2])
    comp_prices = [min(deg_price_space1), min(deg_price_space2)]

    reward_array = revenue_matrix1 + revenue_matrix2
    ind1, ind2 = np.unravel_index(np.argmax(reward_array), np.array(reward_array).shape)
    mono_prices = [price_space1[ind1], price_space2[ind2]] 

    return comp_prices, mono_prices
