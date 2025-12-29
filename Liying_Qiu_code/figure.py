import pickle
import os
import numpy as np
from setup import get_config
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, logistic
from env import *
from utils import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
matplotlib.rc('font',family='Times New Roman')


def plot_comparison(var_per, var_unper, x_label, var_alias, cut_per = 7, cut_unper = 0):
    ks_stats_2side = stats.ks_2samp(var_per, var_unper)
    mw_stats_2side = stats.mannwhitneyu(var_per, var_unper, alternative = "two-sided")

    fig, ax = plt.subplots()
    kde_plot_per = sns.kdeplot(var_per, cumulative = True, common_norm = True, label = "personalized", 
                               color = "black", linestyle = "--", cut = cut_per)
    kde_plot_unper = sns.kdeplot(var_unper, cumulative = True, common_norm = True, label = "unpersonalized", 
                                color = "black", cut = cut_unper)
    ax.set_ylabel("Cumulative density")
    ax.set_xlabel(x_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    plt.legend()
    plt.text(xmin + (xmax - xmin)/100, (ymax + ymin)/2 + (ymax - ymin) * 25/100, 'KS test: statistic = '+ str(np.around(ks_stats_2side.statistic, 3)))
    plt.text(xmin + (xmax - xmin)/100, (ymax + ymin)/2 + (ymax - ymin) * 20/100, 'KS test: p = '+ str(np.around(ks_stats_2side.pvalue, 3)))
    plt.text(xmin + (xmax - xmin)/100, (ymax + ymin)/2 + (ymax - ymin) * 15/100, 'MW test: statistic = '+ str(np.around(mw_stats_2side.statistic, 3)))
    plt.text(xmin + (xmax - xmin)/100, (ymax + ymin)/2 + (ymax - ymin) * 10/100, 'MW test: p = '+ str(np.around(mw_stats_2side.pvalue, 3)))
    plt.savefig(f"./figures/CDF_{var_alias}.pdf")
    plt.show()

def plot_dist_per_unper(args):
    if args.file_prefix is not None and args.file_prefix != 'price-only':
        profit_file_name_per = f'profit_{args.file_prefix}_personalized_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    else:
        if args.num_product == 3:
            profit_file_name_per = f'profit_personalized3_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
            profit_file_name_per = f'profit_personalized_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
            
    if args.file_prefix is not None:
        profit_file_name_unper = f'profit_{args.file_prefix}_unpersonalized2_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    else:
        if args.num_product == 3:
            profit_file_name_unper = f'profit_unpersonalized3_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
            profit_file_name_unper = f'profit_unpersonalized2_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
            
    if args.file_prefix is not None and args.file_prefix != 'price-only':
        if args.do_rational_belief:
            cs_file_name_per = f'cs_rational_belief_{args.file_prefix}_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
            cs_file_name_per = f'cs_{args.file_prefix}_personalized_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    else:
        if args.num_product == 3:
            cs_file_name_per = f'cs_personalized3_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
            if args.do_rational_belief:
                cs_file_name_per = f'cs_rational_belief_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
            else:
                cs_file_name_per = f'cs_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
            
    if args.file_prefix is not None:
        cs_file_name_unper = f'cs_{args.file_prefix}_unpersonalized2_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    else:
        if args.num_product == 3:
            cs_file_name_unper = f'cs_unpersonalized3_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
            cs_file_name_unper = f'cs_unpersonalized2_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    
    try:
        with open(args.data_dir + profit_file_name_per, 'rb') as f:
            profit_data_per = pickle.load(f)
            prices_per = profit_data_per['prices']
            profits_per = profit_data_per['profits']
    except:
        print(f"File {args.data_dir + profit_file_name_per} not found!")
        return
    try:
        with open(args.data_dir + profit_file_name_unper, 'rb') as f:
            profit_data_unper = pickle.load(f)
            prices_unper = profit_data_unper['prices']
            profits_unper = profit_data_unper['profits']
    except:
        print(f"File {args.data_dir + profit_file_name_unper} not found!")
        return
    try:
        with open(args.data_dir + cs_file_name_per, 'rb') as f:
            cs_per = pickle.load(f)
        print(f"File {cs_file_name_per} found.")
        cs_per = np.array(cs_per).reshape(100, 1000).mean(axis = 1)
    except:
        print(f"File {cs_file_name_per} not found.")
    try:
        with open(args.data_dir + cs_file_name_unper, 'rb') as f:
            cs_unper = pickle.load(f)
        print(f"File {cs_file_name_unper} found.")
        cs_unper = np.array(cs_unper).reshape(100, 1000).mean(axis = 1)
    except:
        print(f"File {cs_file_name_unper} not found.")
    
    plot_comparison(cs_per, cs_unper, "Consumer surplus", "cs", 10, 2)
    plot_comparison(prices_per, prices_unper, "Price", "price", 3, 11)
    plot_comparison(profits_per, profits_unper, "Profit", "profit", 3, 11)

def gen_dev_oppo_mean_price(comp_prices, mono_prices, Qs, last_actions, args, dev_index):
    env, agents = make_env(comp_prices, mono_prices, args)
    for i in range(args.num_product):
        agent = agents[i]
        agent.Q = Qs[i]
    unique_actions, counts = np.unique(last_actions, axis = 0, return_counts = True)
    weights = counts/args.n_sum

    dev_price_array = []
    opponent_price_array = []    
    diff_dev_price_array = []
    diff_opponent_price_array = []
    for i in range(len(unique_actions)):
        action = unique_actions[i]
        dev_price_list = []
        opponent_price_list = []
        opponent_price = action[1 - dev_index]
        dev_price = agents[dev_index].action_space[1]
        dev_agent = agents[dev_index]
        opponent_agent = agents[1 - dev_index]

        for t in range(args.n_steps):
            if dev_index == 0:
                state = [dev_price, opponent_price]
            else:
                state = [opponent_price, dev_price]
            dev_price_list.append(dev_price)
            opponent_price_list.append(opponent_price)
            state_index = env.state_space.index(state)
            dev_price = dev_agent.action_space[np.argmax(dev_agent.Q[state_index])]
            opponent_price = opponent_agent.action_space[np.argmax(opponent_agent.Q[state_index])]
            
        dev_price_list = [action[dev_index]] + dev_price_list
        opponent_price_list = [action[1 - dev_index]] + opponent_price_list
        dev_price_array.append(np.array(dev_price_list) * weights[i])
        opponent_price_array.append(np.array(opponent_price_list) * weights[i])
        diff_dev_price_array.append((np.array(dev_price_list) - dev_price_list[0]) *  weights[i])
        diff_opponent_price_array.append((np.array(opponent_price_list) - opponent_price_list[0]) * weights[i])
    return np.sum(dev_price_array, axis = 0), np.sum(opponent_price_array, axis = 0), \
        np.sum(diff_dev_price_array, axis = 0), np.sum(diff_opponent_price_array, axis = 0)

def plot_deviation(args):
    if args.file_prefix is not None and 'UCB' not in args.file_prefix:
        if args.file_prefix == 'sales':
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
    
    summary_file_name = f'./data/deviation_{args.ranking_type}.pkl'
    if os.path.exists(summary_file_name):
        with open(summary_file_name, 'rb') as f:
            results = pickle.load(f)
        mean_dev_all_dev0 = results['mean_dev_all_dev0']
        mean_oppo_all_dev0 = results['mean_oppo_all_dev0']
        diff_mean_dev_all_dev0 = results['diff_mean_dev_all_dev0']
        diff_mean_oppo_all_dev0 = results['diff_mean_oppo_all_dev0']
        mean_dev_all_dev1 = results['mean_dev_all_dev1']
        mean_oppo_all_dev1 = results['mean_oppo_all_dev1']
        diff_mean_dev_all_dev1 = results['diff_mean_dev_all_dev1']
        diff_mean_oppo_all_dev1 = results['diff_mean_oppo_all_dev1']
        print(f"File {summary_file_name} found.")
    else:
        print(f"File {summary_file_name} not found. Generating the file.")
        mean_dev_all_dev0 = []
        mean_oppo_all_dev0 = []
        diff_mean_dev_all_dev0 = []
        diff_mean_oppo_all_dev0 = []
        mean_dev_all_dev1 = []
        mean_oppo_all_dev1 = []
        diff_mean_dev_all_dev1 = []
        diff_mean_oppo_all_dev1 = []
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
            Qs = results['Qs']
            actions = results['actions'][-args.n_sum:]

            mean_dev, mean_oppo, diff_dev, diff_oppo = gen_dev_oppo_mean_price(comp_prices, mono_prices, Qs, actions, args, 0)
            mean_dev_all_dev0.append(mean_dev)
            mean_oppo_all_dev0.append(mean_oppo)
            diff_mean_dev_all_dev0.append(diff_dev)
            diff_mean_oppo_all_dev0.append(diff_oppo)
            mean_dev, mean_oppo, diff_dev, diff_oppo = gen_dev_oppo_mean_price(comp_prices, mono_prices, Qs, actions, args, 1)
            mean_dev_all_dev1.append(mean_dev)
            mean_oppo_all_dev1.append(mean_oppo)
            diff_mean_dev_all_dev1.append(diff_dev)
            diff_mean_oppo_all_dev1.append(diff_oppo)
            # print(f"Seed {seed} done.")
        results = {'mean_dev_all_dev0': mean_dev_all_dev0, 'mean_oppo_all_dev0': mean_oppo_all_dev0,
                    'diff_mean_dev_all_dev0': diff_mean_dev_all_dev0, 'diff_mean_oppo_all_dev0': diff_mean_oppo_all_dev0,
                    'mean_dev_all_dev1': mean_dev_all_dev1, 'mean_oppo_all_dev1': mean_oppo_all_dev1,
                    'diff_mean_dev_all_dev1': diff_mean_dev_all_dev1, 'diff_mean_oppo_all_dev1': diff_mean_oppo_all_dev1}
        with open(summary_file_name, 'wb') as f:
            pickle.dump(results, f)
    
    data_dev_all_dev0 = np.array(mean_dev_all_dev0).mean(axis = 0)
    data_oppo_all_dev0 = np.array(mean_oppo_all_dev0).mean(axis = 0)
    diff_mean_dev_all_dev0 = np.array(diff_mean_dev_all_dev0).mean(axis = 0)
    diff_mean_oppo_all_dev0 = np.array(diff_mean_oppo_all_dev0).mean(axis = 0)
    data_dev_all_dev1 = np.array(mean_dev_all_dev1).mean(axis = 0)
    data_oppo_all_dev1 = np.array(mean_oppo_all_dev1).mean(axis = 0)
    diff_mean_dev_all_dev1 = np.array(diff_mean_dev_all_dev1).mean(axis = 0)
    diff_mean_oppo_all_dev1 = np.array(diff_mean_oppo_all_dev1).mean(axis = 0)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(data_dev_all_dev0)), data_dev_all_dev0, '-o', color = "steelblue", label = "Deviation firm")
    ax.plot(np.arange(len(data_oppo_all_dev0)), data_oppo_all_dev0, color = "indianred", label = "Nondeviating firm", marker = "^")
    ax.axhline(y = data_dev_all_dev0[0], linestyle = '-.', color = "steelblue", label = "Long-run price (deviation firm)")
    ax.axhline(y = data_oppo_all_dev0[0], linestyle = '-.', color = "indianred", label = "Long-run price (nondeviating firm)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(f'./figures/deviation_firm0_{args.ranking_type}.pdf')

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(data_dev_all_dev1)), data_dev_all_dev1, '-o', color = "steelblue", label = "Deviation firm")
    ax.plot(np.arange(len(data_oppo_all_dev1)), data_oppo_all_dev1, color = "indianred", label = "Nondeviating firm", marker = "^")
    ax.axhline(y = data_dev_all_dev1[0], linestyle = '-.', color = "steelblue", label = "Long-run price (deviation firm)")
    ax.axhline(y = data_oppo_all_dev1[0], linestyle = '-.', color = "indianred", label = "Long-run price (nondeviating firm)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(f'./figures/deviation_firm1_{args.ranking_type}.pdf')

    data_dev0 = np.array(mean_dev_all_dev0)
    data_oppo0 = np.array(mean_oppo_all_dev0)
    data_dev1 = np.array(mean_dev_all_dev1)
    data_oppo1 = np.array(mean_oppo_all_dev1)
    data_dev0 = data_dev0 - np.mean(data_dev0[:,0])
    data_dev0[:,0] = 0.
    data_oppo0 = data_oppo0 - np.mean(data_oppo0[:,0])
    data_oppo0[:,0:2] = 0.
    data_dev1 = data_dev1 - np.mean(data_dev1[:,0])
    data_dev1[:,0] = 0.
    data_oppo1 = data_oppo1 - np.mean(data_oppo1[:,0])
    data_oppo1[:,0:2] = 0.

    fig, ax = plt.subplots()
    ax.boxplot(data_dev0, 
               patch_artist=True, 
               showfliers=False,
               boxprops=dict(facecolor="sandybrown", color="black"),
               medianprops=dict(color="black", linewidth=1),
               whiskerprops = dict(linestyle='--', linewidth=1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Price change")
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(f'./figures/boxplot_{args.ranking_type}_dev0.pdf')
    plt.show()

    fig, ax = plt.subplots()
    ax.boxplot(data_dev1, 
               patch_artist=True, 
               showfliers=False,
               boxprops=dict(facecolor="sandybrown", color="black"),
               medianprops=dict(color="black", linewidth=1),
               whiskerprops = dict(linestyle='--', linewidth=1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Price change")
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(f'./figures/boxplot_{args.ranking_type}_dev1.pdf')
    plt.show()

    fig, ax = plt.subplots()
    ax.boxplot(data_oppo0, 
               patch_artist=True, 
               showfliers=False,
               boxprops=dict(facecolor="sandybrown", color="black"),
               medianprops=dict(color="black", linewidth=1),
               whiskerprops = dict(linestyle='--', linewidth=1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Price change")
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(f'./figures/boxplot_{args.ranking_type}_oppo0.pdf')
    plt.show()

    fig, ax = plt.subplots()
    ax.boxplot(data_oppo1, 
               patch_artist=True, 
               showfliers=False,
               boxprops=dict(facecolor="sandybrown", color="black"),
               medianprops=dict(color="black", linewidth=1),
               whiskerprops = dict(linestyle='--', linewidth=1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Price change")
    ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(f'./figures/boxplot_{args.ranking_type}_oppo1.pdf')
    plt.show()

def plot_analytical_mechanism(args):
    init_unper_price1, init_unper_price2 = 2.63, 3.05
    init_per_price1, init_per_price2 = 3.41, 3.92
    init_profit_per = gen_per_profit_correct([init_per_price1, init_per_price2], args.a0, args.a, args.mu, args.ps, args.mc)
    _, init_profit_unper = gen_expected_profit_unper_two_product([init_per_price1, init_per_price2], args)
    price_start1 = args.mc[0]
    price_start2 = args.mc[1]

    def gen_profit_diff_per(dev_price_pair, init_profit):
        dev_profit = gen_per_profit_correct(dev_price_pair, args.a0, args.a, args.mu, args.ps, args.mc)
        return np.array(dev_profit) - np.array(init_profit)
        
    def gen_profit_diff_unper(dev_price_pair, init_profit):
        _, dev_profit = gen_expected_profit_unper_two_product(dev_price_pair, args)
        return np.array(dev_profit) - np.array(init_profit)
    
    dev_price_range1 = np.arange(price_start1, init_per_price1, 0.01)
    dev_price_range2 = np.arange(price_start2, init_per_price2, 0.01)
    dev_profit1_per = np.zeros(len(dev_price_range1))
    pun_profit1_per = np.zeros(len(dev_price_range2))
    dev_profit2_per = np.zeros(len(dev_price_range2))
    pun_profit2_per = np.zeros(len(dev_price_range1))
    dev_profit1_unper = np.zeros(len(dev_price_range1))
    pun_profit1_unper = np.zeros(len(dev_price_range2))
    dev_profit2_unper = np.zeros(len(dev_price_range2))
    pun_profit2_unper = np.zeros(len(dev_price_range1))

    # firm 1 deviates
    for i in range(len(dev_price_range1)):
        dev_price_pair = [dev_price_range1[i], init_per_price2]
        dev_profit1_per[i], _ = gen_profit_diff_per(dev_price_pair, init_profit_per)
        dev_profit1_unper[i], _ = gen_profit_diff_unper(dev_price_pair, init_profit_unper)
    # firm 2 punishes
    for i in range(len(dev_price_range2)):
        dev_price_pair = [init_per_price1, dev_price_range2[i]]
        pun_profit1_per[i], _ = gen_profit_diff_per(dev_price_pair, init_profit_per)
        pun_profit1_unper[i], _ = gen_profit_diff_unper(dev_price_pair, init_profit_unper)
    
    fig, ax = plt.subplots()
    ax.plot(dev_price_range1, dev_profit1_per, label = "personalized", color = "black", linestyle="--")
    ax.plot(dev_price_range1, dev_profit1_unper, label = "unpersonalized", color = "black")
    ax.set_xlabel("Deviation price")
    ax.set_ylabel("Profit difference")
    ax.spines[['right', 'top']].set_visible(False)
    ax.axhline(y=0, color= "gray", linestyle='dashdot')
    plt.legend()
    plt.savefig("./figures/mechanism_deviation_firm1.pdf")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(dev_price_range2, pun_profit1_per, label = "personalized", color = "black", linestyle="--")
    ax.plot(dev_price_range2, pun_profit1_unper, label = "unpersonalized", color = "black")
    ax.set_xlabel("Deviation price")
    ax.set_ylabel("Profit difference")
    ax.spines[['right', 'top']].set_visible(False)
    ax.axhline(y=0, color= "gray", linestyle='dashdot')
    plt.legend()
    plt.savefig("./figures/mechanism_punishment_firm1.pdf")
    plt.show()

    # firm 2 deviates
    for i in range(len(dev_price_range2)):
        dev_price_pair = [init_per_price1, dev_price_range2[i]]
        _, dev_profit2_per[i] = gen_profit_diff_per(dev_price_pair, init_profit_per)
        _, dev_profit2_unper[i] = gen_profit_diff_unper(dev_price_pair, init_profit_unper)
    # firm 1 punishes
    for i in range(len(dev_price_range1)):
        dev_price_pair = [dev_price_range1[i], init_per_price2]
        _, pun_profit2_per[i] = gen_profit_diff_per(dev_price_pair, init_profit_per)
        _, pun_profit2_unper[i] = gen_profit_diff_unper(dev_price_pair, init_profit_unper)
    
    fig, ax = plt.subplots()
    ax.plot(dev_price_range2, dev_profit2_per, label = "personalized", color = "black", linestyle="--")
    ax.plot(dev_price_range2, dev_profit2_unper, label = "unpersonalized", color = "black")
    ax.set_xlabel("Deviation price")
    ax.set_ylabel("Profit difference")
    ax.spines[['right', 'top']].set_visible(False)
    ax.axhline(y=0, color= "gray", linestyle='dashdot')
    plt.legend()
    plt.savefig("./figures/mechanism_deviation_firm2.pdf")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(dev_price_range1, pun_profit2_per, label = "personalized", color = "black", linestyle="--")
    ax.plot(dev_price_range1, pun_profit2_unper, label = "unpersonalized", color = "black")
    ax.set_xlabel("Deviation price")
    ax.set_ylabel("Profit difference")
    ax.spines[['right', 'top']].set_visible(False)
    ax.axhline(y=0, color= "gray", linestyle='dashdot')
    plt.legend()
    plt.savefig("./figures/mechanism_punishment_firm2.pdf")
    plt.show()

def plot_indirect_price_effect(args):
    init_unper_price1 = 2.63
    init_unper_price2 = 3.05
    # unper_price = [init_unper_price1, init_unper_price2]
    init_per_price1 = 3.41
    init_per_price2 = 3.92
    per_price = [init_per_price1, init_per_price2]

    # plot the ranking probability (low-quality firm deviates)
    price_space1 = np.linspace(1.0, init_per_price1, 100)
    unper_prob_list = []
    per_prob_list = []
    for price1 in price_space1:
        unper_prob = gen_unper_ranking_prob(args.a, [price1, init_per_price2], args.gamma, "expected-utility", args=args)
        unper_prob_list.append(unper_prob[0])
        per_prob = logistic.cdf(args.a[0] - price1 - args.a[1] + init_per_price2)
        per_prob_list.append(per_prob)
    
    fig, ax = plt.subplots()
    ax.plot(price_space1, unper_prob_list, label = "unpersonalized", color = "black")
    ax.plot(price_space1, per_prob_list, label = "personalized", color = "black", linestyle="--")
    ax.set_xlabel("Price of low-quality firm", fontsize = 20)
    ax.set_ylabel("Probability of being ranked on the top", fontsize = 20)
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(fontsize = 20)
    plt.savefig("./figures/price_indirect_effect_ranking.pdf")

    # plot the ranking probability (high-quality firm deviates)
    price_space2 = np.linspace(1.5, 4.5, 100)
    unper_prob_list = []
    per_prob_list = []
    for price2 in price_space2:
        unper_prob = gen_unper_ranking_prob(args.a, [init_per_price1, price2], args.gamma, "expected-utility", args=args)
        unper_prob_list.append(unper_prob[1])
        per_prob = logistic.cdf(args.a[1] - price2 - args.a[0] + init_per_price1)
        per_prob_list.append(per_prob)
    
    fig, ax = plt.subplots()
    ax.plot(price_space2, unper_prob_list, label = "unpersonalized", color = "black")
    ax.plot(price_space2, per_prob_list, label = "personalized", color = "black", linestyle="--")
    ax.set_xlabel("Price of high-quality firm", fontsize = 20)
    ax.set_ylabel("Probability of being ranked on the top", fontsize = 20)
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(fontsize = 20)
    plt.savefig("./figures/price_indirect_effect_ranking_high.pdf")

def plot_direct_price_effect(args):
    init_unper_price1 = 2.63
    init_unper_price2 = 3.05
    # unper_price = [init_unper_price1, init_unper_price2]
    init_per_price1 = 3.41
    init_per_price2 = 3.92
    per_price = [init_per_price1, init_per_price2]

    price_space1 = np.linspace(1.0, init_per_price1, 100)
    unper_demand_list = []
    per_demand_list = []
    for price1 in price_space1:
        rank_prob = gen_unper_ranking_prob(args.a, [price1, init_per_price2], args.gamma, "expected-utility", args=args)
        demand = gen_demand_unper_two_product([price1, init_per_price2], args.a0, args.a, args.mu, args.ps, args.sc, rank_prob[0])
        unper_demand_list.append(demand[0])
        demand = gen_correct_demand([price1, init_per_price2], args.a0, args.a, args.mu, args.ps)
        per_demand_list.append(demand[0])
    
    fig, ax = plt.subplots()
    ax.plot(price_space1, unper_demand_list, label = "unpersonalized", color = "black")
    ax.plot(price_space1, per_demand_list, label = "personalized", color = "black", linestyle="--")
    ax.set_xlabel("Price of low-quality firm", fontsize = 20)
    ax.set_ylabel("Demand of low-quality firm", fontsize = 20)
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(fontsize = 20)
    plt.savefig("./figures/price_direct_effect_demand.pdf")

    price_space2 = np.linspace(1.5, 4.5, 100)
    unper_demand_list = []
    per_demand_list = []
    for price2 in price_space2:
        rank_prob = gen_unper_ranking_prob(args.a, [init_per_price1, price2], args.gamma, "expected-utility", args=args)
        demand = gen_demand_unper_two_product([init_per_price1, price2], args.a0, args.a, args.mu, args.ps, args.sc, rank_prob[0])
        unper_demand_list.append(demand[1])
        demand = gen_correct_demand([init_per_price1, price2], args.a0, args.a, args.mu, args.ps)
        per_demand_list.append(demand[1])
    
    fig, ax = plt.subplots()
    ax.plot(price_space2, unper_demand_list, label = "unpersonalized", color = "black")
    ax.plot(price_space2, per_demand_list, label = "personalized", color = "black", linestyle="--")
    ax.set_xlabel("Price of high-quality firm", fontsize = 20)
    ax.set_ylabel("Demand of high-quality firm", fontsize = 20)
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend(fontsize = 20)
    plt.savefig("./figures/price_direct_effect_demand_high.pdf")
    # plt.show()

def plot_lr_heatmap():
    alpha_list = [0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1]
    beta_list = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5]
    cs_table = np.zeros((len(alpha_list), len(beta_list)))
    price_table = np.zeros((len(alpha_list), len(beta_list)))
    profit_table = np.zeros((len(alpha_list), len(beta_list)))
    for i in range(len(alpha_list)):
        for j in range(len(beta_list)):
            alpha = alpha_list[i]
            beta = beta_list[j]
            per_cs_file = f'cs_rational_belief_personalized_n2_a00.0_a[4.0, 4.5]_mc[1.0, 1.5]_gamma-1.0_sc1.5_alpha{alpha}_beta{beta}_delta0.95.pkl'
            with open(f'./data/{per_cs_file}', 'rb') as f:
                per_cs = pickle.load(f)

            unper_cs_file = f'cs_unpersonalized2_n2_a00.0_a[4.0, 4.5]_mc[1.0, 1.5]_gamma-1.0_sc1.5_alpha{alpha}_beta{beta}_delta0.95.pkl'
            with open(f'./data_psc/data/{unper_cs_file}', 'rb') as f:
                unper_cs = pickle.load(f)
            
            per_cs = np.mean(per_cs)
            unper_cs = np.mean(unper_cs)
            # print(f"alpha: {alpha}, beta: {beta}, per_cs: {per_cs}, unper_cs: {unper_cs}")
            diff = (per_cs - unper_cs)/unper_cs * 100
            cs_table[i, j] = diff

            per_profit_file = f'profit_personalized_n2_a00.0_a[4.0, 4.5]_mc[1.0, 1.5]_gamma-1.0_sc1.5_alpha{alpha}_beta{beta}_delta0.95.pkl'
            with open(f'./data_psc/data/{per_profit_file}', 'rb') as f:
                per_data = pickle.load(f)
            
            unper_profit_file = f'profit_unpersonalized2_n2_a00.0_a[4.0, 4.5]_mc[1.0, 1.5]_gamma-1.0_sc1.5_alpha{alpha}_beta{beta}_delta0.95.pkl'
            with open(f'./data_psc/data/{unper_profit_file}', 'rb') as f:
                unper_data = pickle.load(f)
            
            per_price = per_data['prices']
            unper_price = unper_data['prices']
            per_profit = per_data['profits']
            unper_profit = unper_data['profits']

            per_price = np.mean(per_price)
            unper_price = np.mean(unper_price)
            per_profit = np.mean(per_profit)
            unper_profit = np.mean(unper_profit)

            price_table[i, j] = (per_price - unper_price)/unper_price * 100
            profit_table[i, j] = (per_profit - unper_profit)/unper_profit * 100

    sns.set()
    fig, ax = plt.subplots()
    sns.heatmap(cs_table, annot=True, fmt=".0f", cmap="Blues", ax=ax, vmin=-30, vmax=-5,
                xticklabels=beta_list, yticklabels=alpha_list)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig("./figures/lr_heatmap_cs.pdf")

    def custom_annot(val):
        val = float(val)
        return f"+{val:.0f}" if val >= 0 else f"{val:.0f}"
    
    # price_annot = np.vectorize(custom_annot)(price_table)
    # print(price_annot)
    fig, ax = plt.subplots()
    sns.heatmap(price_table, annot=True, cmap="Reds", ax=ax, vmin=15, vmax=85,
                xticklabels=beta_list, yticklabels=alpha_list)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    plt.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.savefig("./figures/lr_heatmap_price.pdf")

    fig, ax = plt.subplots()
    sns.heatmap(profit_table, annot=True, cmap="Reds", ax=ax, vmin=15, vmax=85,
                xticklabels=beta_list, yticklabels=alpha_list)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig("./figures/lr_heatmap_profit.pdf")

def plot_accuracy():
    baseline_cs_file_name = 'cs_unpersonalized2_n2_a00.0_a[4.0, 4.5]_mc[1.0, 1.5]_gamma-1.0_sc1.5_alpha0.1_beta2e-06_delta0.95.pkl'
    with open(f'./data/{baseline_cs_file_name}', 'rb') as f:
        baseline_cs = pickle.load(f)
    baseline_cs = np.array(baseline_cs).reshape(100, 1000).mean(axis = 1).mean()

    baseline_profit_file_name = 'profit_unpersonalized2_n2_a00.0_a[4.0, 4.5]_mc[1.0, 1.5]_gamma-1.0_sc1.5_alpha0.1_beta2e-06_delta0.95.pkl'
    with open(f'./data/{baseline_profit_file_name}', 'rb') as f:
        baseline_profit_data = pickle.load(f)
    baseline_profit = np.mean(baseline_profit_data['profits'])
    baseline_price = np.mean(baseline_profit_data['prices'])

    lamb_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cs_list = []
    for lamb in lamb_list:
        cs_file_name = f'cs_lambda{lamb}_imperfect-rational-n_customer5000_n2_a00.0_a[4.0, 4.5]_mc[1.0, 1.5]_gamma-1.0_sc1.5_alpha0.1_beta2e-06_delta0.95.pkl'
        with open(f'./data/{cs_file_name}', 'rb') as f:
            cs = pickle.load(f)
        cs = np.array(cs).reshape(100, 1000).mean(axis = 1).tolist()
        cs_list.append(cs)
    
    profit_list = []
    price_list = []
    for lamb in lamb_list:
        profit_file_name = f'profit_lambda{lamb}_imperfect-rational-n_customer5000_n2_a00.0_a[4.0, 4.5]_mc[1.0, 1.5]_gamma-1.0_sc1.5_alpha0.1_beta2e-06_delta0.95.pkl'
        with open(f'./data/{profit_file_name}', 'rb') as f:
            profit_data = pickle.load(f)
        profit = profit_data['profits']
        profit_list.append(profit)
        price = profit_data['prices']
        price_list.append(price)
    
    fig, ax = plt.subplots()
    ax.boxplot(cs_list,
               patch_artist=True, 
               showfliers=False,
               boxprops=dict(facecolor="sandybrown", color="black"),
               medianprops=dict(color="black", linewidth=1),
               whiskerprops = dict(linestyle='--', linewidth=1))
    ax.set_xticklabels(lamb_list)
    ax.set_ylabel("Consumer surplus")
    ax.set_xlabel("Accuracy")
    ax.spines[['right', 'top']].set_visible(False)
    ax.axhline(y=baseline_cs, color= "r", linestyle='dashed', label = "unpersonalized")
    plt.legend()
    plt.savefig("./figures/box_accuracy_cs.pdf")
    plt.show()

    fig, ax = plt.subplots()
    ax.boxplot(profit_list,
               patch_artist=True, 
               showfliers=False,
               boxprops=dict(facecolor="sandybrown", color="black"),
               medianprops=dict(color="black", linewidth=1),
               whiskerprops = dict(linestyle='--', linewidth=1))
    ax.set_xticklabels(lamb_list)
    ax.set_ylabel("Profit")
    ax.set_xlabel("Accuracy")
    ax.spines[['right', 'top']].set_visible(False)
    ax.axhline(y=baseline_profit, color= "r", linestyle='dashed', label = "unpersonalized")
    plt.legend()
    plt.savefig("./figures/box_accuracy_profit.pdf")
    plt.show()

    fig, ax = plt.subplots()
    ax.boxplot(price_list,
               patch_artist=True, 
               showfliers=False,
               boxprops=dict(facecolor="sandybrown", color="black"),
               medianprops=dict(color="black", linewidth=1),
               whiskerprops = dict(linestyle='--', linewidth=1))
    ax.set_xticklabels(lamb_list)
    ax.set_ylabel("Price")
    ax.set_xlabel("Accuracy")
    ax.spines[['right', 'top']].set_visible(False)
    ax.axhline(y=baseline_price, color= "r", linestyle='dashed', label = "unpersonalized")
    plt.legend()
    plt.savefig("./figures/box_accuracy_price.pdf")
    plt.show()    

if __name__ == "__main__":
    print(f"Start generating figures.")
    args = get_config()

    plot_dist_per_unper(args) # figure1
    plot_deviation(args) # figure2, figure3, and figure4
    plot_analytical_mechanism(args) # figure5
    plot_indirect_price_effect(args) # figure6
    plot_direct_price_effect(args) # figure6
    plot_lr_heatmap() # figure7
    plot_accuracy() # figure8
    
    print("Done!")

