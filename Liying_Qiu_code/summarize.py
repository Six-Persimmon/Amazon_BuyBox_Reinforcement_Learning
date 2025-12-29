import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))
import numpy as np
from scipy.optimize import fsolve
import scipy.special as special
from scipy import integrate
from utils import *
import pickle
import multiprocessing as mp
from setup import get_config

# generate consumer surplus for unpersonalized ranking
@jit_integrand_function
def pdf_no_search_max_utility(integrand_args):
    x, mu, ps, a0, a1, p1 = integrand_args
    alpha = mu * np.log(np.exp(a0/mu) + np.exp((a1 - ps * p1)/mu))
    z = (x - alpha)/mu
    return x * 1/mu * np.exp(-(z + np.exp(-z)))

@jit_integrand_function
def pdf_search_buy_utility(integrand_args):
    x2, x2_hat, mu, ps, a0, a1, a2, p1, p2 = integrand_args
    z_inside = (x2 - (a2 - ps * p2))/mu
    pdf_inside = x2 * 1/mu * np.exp(-(z_inside + np.exp(-z_inside)))
    z_outside = (x2_hat - (mu * np.log(np.exp(a0/mu) + np.exp((a1 - ps * p1)/mu))))/mu
    pdf_outside = 1/mu * np.exp(-(z_outside + np.exp(-z_outside)))
    return pdf_inside * pdf_outside
    
@jit_integrand_function
def pdf_search_not_buy_utility(integrand_args):
    x2_hat, x2, mu, ps, a0, a1, a2, p1, p2 = integrand_args
    z_inside = (x2 - (a2 - ps * p2))/mu
    pdf_inside = 1/mu * np.exp(-(z_inside + np.exp(-z_inside)))
    z_outside = (x2_hat - (mu * np.log(np.exp(a0/mu) + np.exp((a1 - ps * p1)/mu))))/mu
    pdf_outside = x2_hat * 1/mu * np.exp(-(z_outside + np.exp(-z_outside)))
    return pdf_inside * pdf_outside

def gen_cs_unpersonalized(a0, a1, a2, p1, p2, mu, ps, sc):
    b_bound = fsolve(function, [1], args = (sc, mu))[0]
    p_search = np.exp(-np.exp(- (b_bound - (mu * np.log(np.exp(a0/mu) + np.exp((a1 - ps * p1)/mu)) + ps * p2 - a2)/mu)))
    p_search = p_search.item()
    p_no_search = 1 - p_search
    E_search = mu * np.log(np.exp(a0/mu) + np.exp((a1 - ps * p1)/mu) + np.exp((a2 - ps * p2)/mu)) + np.euler_gamma * mu - sc
    E_search_buy = integrate.dblquad(pdf_search_buy_utility, -np.Inf, mu * b_bound + a2 - ps * p2, lambda x: x, np.Inf, args = (mu, ps, a0, a1, a2, p1, p2,))
    E_search_buy = E_search_buy[0]
    E_search_not_buy = integrate.dblquad(pdf_search_not_buy_utility, -np.Inf, mu * b_bound + a2 - ps * p2, lambda x: x, mu * b_bound + a2 - ps * p2, args = (mu, ps, a0, a1, a2, p1, p2,))
    E_search_not_buy = E_search_not_buy[0]
    E_search = E_search_buy + E_search_not_buy
    E_search /= np.exp(-np.exp(-(mu * b_bound + a2 - ps * p2 - mu * np.log(np.exp(a0/mu) + np.exp((a1 - ps * p1)/mu)))/mu))
    E_search -= sc
    E_no_search = integrate.quad(pdf_no_search_max_utility, mu * b_bound + a2 - ps * p2, np.Inf, args = (mu, ps, a0, a1, p1,))
    E_no_search = E_no_search[0]
    E_no_search /= p_no_search
    if p_search == 0:
        cs_num = E_no_search
    else:
        cs_num = p_search * E_search + p_no_search * E_no_search
    return cs_num

# generate consumer surplus for personalized ranking
def b_fct(b, sc, mu):   
    return  - mu * b + mu * np.euler_gamma - mu * special.expi(-np.exp(-b)) - sc

@jit_integrand_function
def pdf_ranking(integrand_args):
    x2_hat, x2, mu, ps, p1, p2, a0, a1, a2 = integrand_args
    alpha_x2 = a2 - ps * p2
    alpha_x2_hat = a1 - ps * p1
    z2 = (x2 - alpha_x2)/mu
    z2_hat = (x2_hat - alpha_x2_hat)/mu
    return 1/mu * np.exp(-(z2 + np.exp(-z2))) * 1/mu * np.exp(-(z2_hat + np.exp(-z2_hat)))

@jit_integrand_function
def pdf_utility_no_purchase(integrand_args):
    u0, u1, u2, mu, ps, p1, p2, a0, a1, a2 = integrand_args
    z0 = (u0 - a0)/mu
    f_u0 = u0 * 1/mu * np.exp(-(z0 + np.exp(-z0)))
    z1 = (u1 - (a1 - ps * p1))/mu
    f_u1 = 1/mu * np.exp(-(z1 + np.exp(-z1)))
    z2 = (u2 - (a2 - ps * p2))/mu
    f_u2 = 1/mu * np.exp(-(z2 + np.exp(-z2)))
    return f_u0 * f_u1 * f_u2

@jit_integrand_function
def pdf_ranking_no_purchase(integrand_args):
    u0, u1, u2, mu, ps, p1, p2, a0, a1, a2 = integrand_args
    z0 = (u0 - a0)/mu
    f_u0 = 1/mu * np.exp(-(z0 + np.exp(-z0)))
    z1 = (u1 - (a1 - ps * p1))/mu
    f_u1 = 1/mu * np.exp(-(z1 + np.exp(-z1)))
    z2 = (u2 - (a2 - ps * p2))/mu
    f_u2 = 1/mu * np.exp(-(z2 + np.exp(-z2)))
    return f_u0 * f_u1 * f_u2

@jit_integrand_function
def pdf_utility_purchase(integrand_args):
    u1, u1_hat, mu, a0, a1, a2, ps, p1, p2 = integrand_args
    z1 = (u1 - (a1 - ps * p1))/mu
    pdf1 = u1 * 1/mu * np.exp(-(z1 + np.exp(-z1)))
    z1_hat = (u1_hat - (mu * np.log(np.exp(a0/mu) + np.exp((a2 - ps * p2)/mu))))/mu
    pdf1_hat = 1/mu * np.exp(-(z1_hat + np.exp(-z1_hat)))
    return pdf1 * pdf1_hat

@jit_integrand_function
def pdf_ranking_purchase(integrand_args):
    u1, u1_hat, mu, a0, a1, a2, ps, p1, p2 = integrand_args
    z1 = (u1 - (a1 - ps * p1))/mu
    pdf1 = 1/mu * np.exp(-(z1 + np.exp(-z1)))
    z1_hat = (u1_hat - (mu * np.log(np.exp(a0/mu) + np.exp((a2 - ps * p2)/mu))))/mu
    pdf1_hat = 1/mu * np.exp(-(z1_hat + np.exp(-z1_hat)))
    return pdf1 * pdf1_hat

def gen_cs_correct_ranking(sc, mu, a0, a1, a2, ps, p1, p2):
    b_bound = fsolve(b_fct, [1], args = (sc, mu))[0]
    p_12 = integrate.dblquad(pdf_ranking, -np.Inf, np.Inf, lambda x: x, np.Inf, args = (mu, ps, p1, p2, a0, a1, a2))
    E_u0_012 = integrate.tplquad(pdf_utility_no_purchase, -np.Inf, np.Inf, lambda x: x, np.Inf, lambda x, y: y, np.Inf, args = (mu, ps, p1, p2, a0, a1, a2,))
    p_012 = integrate.tplquad(pdf_ranking_no_purchase, -np.Inf, np.Inf, lambda x: x, np.Inf, lambda x, y: y, np.Inf, args = (mu, ps, p1, p2, a0, a1, a2,))
    p_search_012 = integrate.tplquad(pdf_ranking_no_purchase, -np.Inf, mu * b_bound + a2 - ps * p2, lambda x: x, mu * b_bound + a2 - ps * p2, lambda x, y: y, mu * b_bound + a2 - ps * p2, args = (mu, ps, p1, p2, a0, a1, a2,))
    E_CS_012 = E_u0_012[0] - p_search_012[0] * sc
    E_u1_012 = integrate.dblquad(pdf_utility_purchase, -np.Inf, np.Inf, lambda x: x, np.Inf, args = (mu, a0, a1, a2, ps, p1, p2,))
    p_1max02 = integrate.dblquad(pdf_ranking_purchase, -np.Inf, np.Inf, lambda x: x, np.Inf, args = (mu, a0, a1, a2, ps, p1, p2, ))
    p_search_1max02 = integrate.dblquad(pdf_ranking_purchase, -np.Inf, mu * b_bound + a2 - ps * p2, lambda x: x, mu * b_bound + a2 - ps * p2, args = (mu, a0, a1, a2, ps, p1, p2, ))
    E_CS_1max02 = E_u1_012[0] - p_search_1max02[0] * sc
    return  E_CS_012 + E_CS_1max02

def gen_cs_personalized(sc, mu, a0, a1, a2, ps, p1, p2):
    E_CS_012_1max02 = gen_cs_correct_ranking(sc, mu, a0, a1, a2, ps, p1, p2)
    E_CS_021_2max01 = gen_cs_correct_ranking(sc, mu, a0, a2, a1, ps, p2, p1)
    cs = E_CS_012_1max02 + E_CS_021_2max01
    return cs

@jit_integrand_function
def pdf_incorrect_ranking(args):
    u0, u2, u1, mu, ps, p1, p2, a0, a1, a2 = args
    z0 = (u0 - a0)/mu
    f_u0 = 1/mu * np.exp(-(z0 + np.exp(-z0)))
    z1 = (u1 - (a1 - ps * p1))/mu
    f_u1 = 1/mu * np.exp(-(z1 + np.exp(-z1)))
    z2 = (u2 - (a2 - ps * p2))/mu
    f_u2 = 1/mu * np.exp(-(z2 + np.exp(-z2)))
    return f_u0 * f_u1 * f_u2

@jit_integrand_function
def pdf_utility_purchase_with_search(args):
    u1, u1_hat, mu, a0, a1, a2, ps, p1, p2 = args
    z1 = (u1 - (a1 - ps * p1))/mu
    pdf1 = u1 * 1/mu * np.exp(-(z1 + np.exp(-z1)))
    z1_hat = (u1_hat - (mu * np.log(np.exp(a0/mu) + np.exp((a2 - ps * p2)/mu))))/mu
    pdf1_hat = 1/mu * np.exp(-(z1_hat + np.exp(-z1_hat)))
    return pdf1 * pdf1_hat

@jit_integrand_function
def pdf_utility_purchase_without_search(args):
    u1_hat, mu, a0, a1, a2, ps, p1, p2 = args
    cdf1 = np.exp(-np.exp(-(u1_hat - (a1 - ps * p1))/mu))
    z1_hat = (u1_hat - (mu * np.log(np.exp(a0/mu) + np.exp((a2 - ps * p2)/mu))))/mu
    pdf1_hat = u1_hat * 1/mu * np.exp(-(z1_hat + np.exp(-z1_hat)))
    return (1 - cdf1) * pdf1_hat

def gen_cs_incorrect_ranking(sc, mu, a0, a1, a2, ps, p1, p2):
    b_bound = fsolve(b_fct, [1], args = (sc, mu))[0]
    E_u0_012 = integrate.tplquad(pdf_utility_no_purchase, -np.Inf, np.Inf, lambda x: x, np.Inf, lambda x, y: y, np.Inf, args = (mu, ps, p1, p2, a0, a1, a2,))    
    p_search_012 = integrate.tplquad(pdf_incorrect_ranking, -np.Inf, mu * b_bound + a1 - ps * p1, lambda x: x, mu * b_bound + a1 - ps * p1, lambda x, y: y, mu * b_bound + a1 - ps * p1, args = (mu, ps, p1, p2, a0, a1, a2,))    
    E_CS_012 = E_u0_012[0] - p_search_012[0] * sc
    E_CS_1max02_nosearch = integrate.quad(pdf_utility_purchase_without_search, mu * b_bound + a1 - ps * p1, np.Inf, args = (mu, a0, a1, a2, ps, p1, p2,))
    E_CS_1max02_nosearch = E_CS_1max02_nosearch[0]
    p_search_1max02 = integrate.dblquad(pdf_ranking_purchase, -np.Inf, mu * b_bound + a1 - ps * p1, lambda x: x, np.Inf, args = (mu, a0, a1, a2, ps, p1, p2, ))
    E_CS_1max02_search = integrate.dblquad(pdf_utility_purchase_with_search, -np.Inf, mu * b_bound + a1 - ps * p1, lambda x: x, np.Inf, args = (mu, a0, a1, a2, ps, p1, p2,))
    E_CS_1max02_search = E_CS_1max02_search[0] - p_search_1max02[0] * sc
    return E_CS_012 + E_CS_1max02_search + E_CS_1max02_nosearch

def gen_cs_imperfect(p1, p2, args):
    cs_num_correct1 = gen_cs_correct_ranking(args.sc, args.mu, args.a0, args.a[0], args.a[1], args.ps, p1, p2)
    cs_num_correct2 = gen_cs_correct_ranking(args.sc, args.mu, args.a0, args.a[1], args.a[0], args.ps, p2, p1)
    cs_num_incorrect1 = gen_cs_incorrect_ranking(args.sc, args.mu, args.a0, args.a[0], args.a[1], args.ps, p1, p2)
    cs_num_incorrect2 = gen_cs_incorrect_ranking(args.sc, args.mu, args.a0, args.a[1], args.a[0], args.ps, p2, p1)
    cs = args.lamb * (cs_num_correct1 + cs_num_correct2) + (1 - args.lamb) * (cs_num_incorrect1 + cs_num_incorrect2)
    return cs

def gen_cs_personalized_purchase(price, args):
    '''
    This function calculates the purchase utility, which is the consumer surplus when consumer never searches the second product.
    It is essentially the expected utility if consumers purchase the first product.
    '''
    num_product = args.num_product
    assert num_product == len(args.a), "The number of products should be the same as the length of a."
    var = np.exp(args.a0/args.mu)
    for i in range(num_product):
        var += np.exp((args.a[i] - args.ps * price[i])/args.mu)
    return np.log(var) + np.euler_gamma

def search_cs_by_action(cs_spaces, state_space, action):
    return cs_spaces[state_space.index(action)]

def output_consumer_surplus(args):
    if args.file_prefix is not None:
        if args.do_rational_belief:
            cs_file_name = f'cs_rational_belief_{args.file_prefix}_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
            cs_file_name = f'cs_{args.file_prefix}_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    else:
        if args.do_rational_belief:
            cs_file_name = f'cs_rational_belief_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
            cs_file_name = f'cs_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    
    if not os.path.exists(args.data_dir + cs_file_name) or args.do_rewrite or args.do_rewrite_cs:
        state_space = None
        cs_list = []
        for seed in range(args.start_seed, args.end_seed):
            if args.file_prefix is not None:
                rl_file_name = f'{args.file_prefix}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}_seed{seed}.pkl'
            else:
                rl_file_name = f'n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}_seed{seed}.pkl'
            with open(args.output_dir + args.ranking_type + '/' + args.learning_algo + '/' + rl_file_name, 'rb') as f:
                results = pickle.load(f)
            
            if not state_space:
                print("Generating consumer surplus space.")
                state_space = results['state_space']
                if args.ranking_type == "personalized":
                    if args.do_rational_belief:
                        pool = mp.Pool(mp.cpu_count()-1)
                        cs_spaces = pool.starmap(gen_cs_personalized_purchase, [(state, args) for state in state_space])
                        pool.close()
                    else:
                        if args.score_type == "risk-averse":
                            if args.do_rational_belief:
                                pool = mp.Pool(mp.cpu_count()-1)
                                cs_spaces = pool.starmap(simulate_cs_rational_belief_personalized2_risk_utility, [(state, args) for state in state_space])
                                pool.close()
                            else:
                                pool = mp.Pool(mp.cpu_count()-1)
                                cs_spaces = pool.starmap(simulate_demand_personalized2_risk_utility, [(state, args, 0, True) for state in state_space])
                                pool.close()
                        else:
                            pool = mp.Pool(mp.cpu_count()-1)
                            cs_spaces = pool.starmap(gen_cs_personalized, [(args.sc, args.mu, args.a0, args.a[0], args.a[1], args.ps, state[0], state[1]) for state in state_space])
                            pool.close()
                elif args.ranking_type == "unpersonalized2":
                    if args.score_type == 'sales':
                        pass
                    elif args.score_type == 'risk-averse':
                        pool = mp.Pool(mp.cpu_count()-1)
                        cs_spaces = pool.starmap(simulate_demand_unpersonalized2_risk_utility, [(state, args, 0, None, True) for state in state_space])
                        pool.close()
                    else:
                        cs_spaces = []
                        for state in state_space:
                            rank_prob = gen_unper_ranking_prob(args.a, state, args.gamma)[0]
                            if rank_prob == 1:
                                cs_tmp = gen_cs_unpersonalized(args.a0, args.a[0], args.a[1], state[0], state[1], args.mu, args.ps, args.sc)
                            elif rank_prob == 0:
                                cs_tmp = gen_cs_unpersonalized(args.a0, args.a[1], args.a[0], state[1], state[0], args.mu, args.ps, args.sc)
                            cs_spaces.append(cs_tmp)
                elif args.ranking_type == "unpersonalized3":
                    gen_consumer_surplus = True
                    seed = 0
                    rank_space = []
                    for state in state_space:
                        rank_space.append(gen_unper_ranking_unper3(args.a, state, args.gamma))
                    pool = mp.Pool(mp.cpu_count()-1)
                    cs_spaces = pool.starmap(simulate_demand_unpersonalized3, [(state_space[i], rank_space[i], args, seed, gen_consumer_surplus) for i in range(len(state_space))])
                    pool.close()
                elif args.ranking_type == "personalized3":
                    if args.do_rational_belief:
                        pool = mp.Pool(mp.cpu_count()-1)
                        cs_spaces = pool.starmap(gen_cs_personalized_purchase, [(state, args) for state in state_space])
                        pool.close()
                    else:
                        pool = mp.Pool(mp.cpu_count()-1)
                        cs_spaces = pool.starmap(simulate_cs_personalized3, [(state, args) for state in state_space])
                        pool.close()
                elif args.ranking_type == "imperfect":
                    pool = mp.Pool(mp.cpu_count()-1)
                    cs_spaces = pool.starmap(gen_cs_imperfect, [(state[0], state[1], args) for state in state_space])
                    pool.close()
                elif args.ranking_type == "imperfect-rational" or args.ranking_type == "imperfect-rational-n_customer5000":
                    pool = mp.Pool(mp.cpu_count()-1)
                    cs_spaces = pool.starmap(simulate_imperfect_per_rational, [(state, args, 0, True) for state in state_space])
                    pool.close()

            actions = results['actions'][-args.n_sum:]

            if args.score_type == 'sales':
                ranks = results['states'][1][-args.n_sum:]
                for i in range(len(actions)):
                    rank = ranks[i]
                    p1, p2 = actions[i]
                    if rank == 1:
                        cs = gen_cs_unpersonalized(args.a0, args.a[0], args.a[1], p1, p2, args.mu, args.ps, args.sc)
                    elif rank == 0:
                        cs = gen_cs_unpersonalized(args.a0, args.a[1], args.a[0], p2, p1, args.mu, args.ps, args.sc)
                    cs_list.append(cs)
            else:
                for action in actions:
                    cs_list.append(cs_spaces[state_space.index(action)])

        with open(args.data_dir + cs_file_name, 'wb') as f:
            pickle.dump(cs_list, f)
    else:
        print(f"File {cs_file_name} already exists.")
    
def output_profit_gain(args):
    if args.file_prefix is not None and 'UCB' not in args.file_prefix:
        if args.file_prefix == 'sales' or args.file_prefix == 'sales-kappa0.25':
            assert args.score_type == 'sales'
            file_name = f'init_{args.file_prefix}_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
_{args.step_size_init}_nactions{args.n_action}_delta{args.delta}.pkl'
        else:
            file_name = f'init_{args.file_prefix}_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    else:
        file_name = f'init_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    with open(args.data_dir + file_name, 'rb') as f:
        init_data = pickle.load(f)
    
    comp_prices = init_data['comp_prices']
    mono_prices = init_data['mono_prices']
    if args.file_prefix == 'sales' or args.file_prefix == 'sales-kappa0.25' or args.score_type == 'risk-averse' or args.ranking_type == 'imperfect-rational' or args.ranking_type == 'imperfect-rational-n_customer5000':
        comp_profit = None
        mono_profit = None
    else:
        comp_profit = np.mean(init_data['comp_profit'])
        mono_profit = init_data['mono_profit']/2

    if args.file_prefix is not None:
        profit_file_name = f'profit_{args.file_prefix}_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    else:
        profit_file_name = f'profit_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    if not os.path.exists(args.data_dir + profit_file_name) or args.do_rewrite or args.do_rewrite_profit_gain:
        if args.do_rewrite:
            print(f"Rewriting profit gain.")
        
        prices = []
        profits = []
        price_gains = []
        profit_gains = []
        for seed in range(args.start_seed, args.end_seed):
            if args.file_prefix is not None:
                rl_file_name = f'{args.file_prefix}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}_seed{seed}.pkl'
            else:
                rl_file_name = f'n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}_seed{seed}.pkl'
                
            if not os.path.exists(args.output_dir + args.ranking_type + '/' + args.learning_algo + '/' + rl_file_name):
                print(f"File {rl_file_name} does not exist.")
                break
            with open(args.output_dir + args.ranking_type + '/' + args.learning_algo + '/' + rl_file_name, 'rb') as f:
                results = pickle.load(f)
            rewards =  results['rewards'][-args.n_sum:]
            actions = results['actions'][-args.n_sum:]
            prices.append(np.mean(actions))
            profits.append(np.mean(rewards))
            price_gains.append((np.mean(actions) - np.mean(comp_prices))/(np.mean(mono_prices) - np.mean(comp_prices)))
            if args.file_prefix == 'sales' or args.file_prefix == 'sales-kappa0.25' or args.score_type == 'risk-averse' or args.ranking_type == 'imperfect-rational' or args.ranking_type == 'imperfect-rational-n_customer5000':
                profit_gains.append(None)
            else:
                profit_gains.append((np.mean(rewards) - comp_profit)/(mono_profit - comp_profit))
        
        with open(args.data_dir + profit_file_name, 'wb') as f:
            pickle.dump({'prices': prices, 'profits': profits, 'price_gains': price_gains, 'profit_gains': profit_gains}, f)
    else:
        print(f"File {profit_file_name} already exists.")

def check_cs_personalized(args):
    p1_space = np.linspace(1, 4, 10)
    p2_space = np.linspace(1.5, 4.5, 10)

    cs_nums = []
    cs_olds = []
    for p1 in p1_space:
        for p2 in p2_space:
            cs_num = gen_cs_personalized_purchase([p1, p2], args)
            cs_nums.append(cs_num)
            cs_old = gen_cs_personalized(args.sc, args.mu, args.a0, args.a[0], args.a[1], args.ps, p1, p2)
            cs_olds.append(cs_old)
            print(f"Finishing p1 = {p1}, p2 = {p2}.")
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(100), cs_nums, label='New')
    ax.plot(np.arange(100), cs_olds, label='Old')
    ax.legend()
    plt.show()
    
if __name__ == '__main__':
    args = get_config()
    output_profit_gain(args)
    output_consumer_surplus(args)