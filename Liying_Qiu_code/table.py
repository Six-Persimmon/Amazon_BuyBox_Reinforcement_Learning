import pickle
import numpy as np
from scipy.stats import kstest, mannwhitneyu, logistic
from setup import get_config
from utils import *

def gen_stat_var(per_var, unper_var):
    ks = kstest(per_var, unper_var, alternative = "two-sided")
    ks_p = ks.pvalue
    ks = np.around(ks.statistic, 3)
    if ks_p < 0.01:
        ks = f"{ks}***"
    elif ks_p < 0.05:
        ks = f"{ks}**"
    elif ks_p < 0.1:
        ks = f"{ks}*"
    mw = mannwhitneyu(per_var, unper_var)
    mw_p = mw.pvalue
    mw = np.around(mw.statistic, 3)
    if mw_p < 0.01:
        mw = f"{mw}***"
    elif mw_p < 0.05:
        mw = f"{mw}**"
    elif mw_p < 0.1:
        mw = f"{mw}*"
    perc_change = int(100 * (np.mean(per_var) - np.mean(unper_var))/np.mean(unper_var))
    if perc_change > 0:
        perc_change = f"+{perc_change}"
    return ks, mw, perc_change

def gen_summary_table(args):
    if args.file_prefix is not None:
        if args.file_prefix == 'sales' or args.file_prefix == 'sales-kappa0.25':
            profit_file_name_per = f'profit_personalized_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
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
            
    if args.file_prefix is not None:
        if args.file_prefix == 'sales' or args.file_prefix == 'sales-kappa0.25':
            if args.do_rational_belief:
                cs_file_name_per = f'cs_rational_belief_personalized_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
            else:
                cs_file_name_per = f'cs_personalized_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
            if args.do_rational_belief:
                cs_file_name_per = f'cs_rational_belief_{args.file_prefix}_{args.ranking_type}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
            else:
                cs_file_name_per = f'cs_{args.file_prefix}_personalized_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
    else:
        if args.do_rational_belief:
            if args.num_product == 3:
                cs_file_name_per = f'cs_rational_belief_personalized3_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
            else:
                cs_file_name_per = f'cs_rational_belief_personalized_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
        else:
            if args.num_product == 3:
                cs_file_name_per = f'cs_personalized3_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
    alpha{args.alpha}_beta{args.beta}_delta{args.delta}.pkl'
            else:
                cs_file_name_per = f'cs_personalized_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
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
    
    print(f"Profit file name personalized: {profit_file_name_per}")
    print(f"Profit file name unpersonalized: {profit_file_name_unper}")
    print(f"CS file name personalized: {cs_file_name_per}")
    print(f"CS file name unpersonalized: {cs_file_name_unper}")

    try:
        with open(args.data_dir + profit_file_name_per, 'rb') as f:
            profit_data_per = pickle.load(f)
            prices_per = profit_data_per['prices']
            profits_per = profit_data_per['profits']
        print(f"File {profit_file_name_per} found.")
        print(f"Prices: {np.mean(prices_per)}")
        print(f"Profits: {np.mean(profits_per)}")
    except:
        print(f"File {profit_file_name_per} not found.")
    try:
        with open(args.data_dir + profit_file_name_unper, 'rb') as f:
            profit_data_unper = pickle.load(f)
            prices_unper = profit_data_unper['prices']
            profits_unper = profit_data_unper['profits']
        print(f"File {profit_file_name_unper} found.")
        print(f"Prices: {np.mean(prices_unper)}")
        print(f"Profits: {np.mean(profits_unper)}")
    except:
        print(f"File {profit_file_name_unper} not found.")
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
    
    prices_ks, prices_mw, price_perc_change = gen_stat_var(prices_per, prices_unper)
    profits_ks, profts_mw, profit_perc_change = gen_stat_var(profits_per, profits_unper)
    cs_ks, cs_mw, cs_perc_change = gen_stat_var(cs_per, cs_unper)

    sum_tex = rf"""\begin{{table}}[h]
\centering
\begin{{threeparttable}}
\centering
\caption{{{args.table_caption}}}
\begin{{tabular}}{{cccccc}}
\hline
  & Unpersonalized  &  Personalized  & \% Change & KS & MW \\\hline
Q-learning price & {round(np.mean(prices_unper), 2)} & {round(np.mean(prices_per), 2)} & {price_perc_change}\% & {prices_ks} & {prices_mw} \\
& ({round(np.std(prices_unper), 2)}) & ({round(np.std(prices_per), 2)}) & \\
Profit & {round(np.mean(profits_unper), 2)} & {round(np.mean(profits_per), 2)} & {profit_perc_change}\% & {profits_ks} & {profts_mw} \\
& ({round(np.std(profits_unper), 2)}) & ({round(np.std(profits_per), 2)}) & & & \\
Consumer surplus & {round(np.mean(cs_unper), 2)} & {round(np.mean(cs_per), 2)} & {cs_perc_change}\% & {cs_ks} & {cs_mw} \\
& ({round(np.std(cs_unper), 2)}) & ({round(np.std(cs_per), 2)}) & \vspace{{0.15cm}} \\ 
 \hline
\end{{tabular}}
\label{{table:{args.table_label}}}
\begin{{tablenotes}}
  \footnotesize
  \item Note: The parameter set is: $a_0 = {args.a0}, a_1 = {args.a[0]}, a_2 = {args.a[1]}, \rho = {args.sc}, \phi = {args.ps}, \mu = {args.mu}, mc_1 = {args.mc[0]}, mc_2 = {args.mc[1]}, \
n = {args.num_product}, \lambda = {args.lamb}, \gamma = {args.gamma}, \alpha = {args.alpha}, \beta = {args.beta}$. All variables are averaged over the last 1000 steps. \
KS indicates the statistics for the Kolmogorov-Smirnov test. MW indicates the statistics for the Mann-Whitney U rank test.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}"""

    if args.file_prefix is not None:
        tex_file = f'compare_{args.file_prefix}_per_unper_gamma{args.gamma}'
    else:
        tex_file = f'compare_per_unper_gamma{args.gamma}'
    
    if args.table_name is not None:
        output_name = './tables/' + tex_file + args.table_name + '.tex'
    else:
        output_name = './tables/' + tex_file + '.tex'
    
    print(f"Writing to {output_name}")
    with open(output_name, 'w') as f:
        f.write(sum_tex)

def gen_price_elasticity(args):
    init_per_price1 = 3.41
    init_per_price2 = 3.92
    per_price = [init_per_price1, init_per_price2]

    original_rank_prob, profit = gen_expected_profit_unper_two_product(per_price, args)
    if original_rank_prob[0] == 0:
        d1, d2 = gen_demand_unper_two_product(per_price, args.a0, args.a, args.mu, args.ps, args.sc, 0)
    elif original_rank_prob[0] == 1:
        d1, d2 = gen_demand_unper_two_product(per_price, args.a0, args.a, args.mu, args.ps, args.sc, 1)
    
    original_per_rank_prob1 = logistic.cdf(args.a[0] - init_per_price1 - args.a[1] + init_per_price2)
    original_per_rank_prob2 = logistic.cdf(args.a[1] - init_per_price2 - args.a[0] + init_per_price1)
    original_demand = gen_correct_demand(per_price, args.a0, args.a, args.mu, args.ps)

    print("Calculating the direct price effect on demand for unpersonalized ranking.")
    unper_rank_prob = gen_unper_ranking_prob(args.a, [init_per_price1, init_per_price2-0.1], args.gamma, "expected-utility", args=args)
    demand_new = gen_demand_unper_two_product([init_per_price1, init_per_price2-0.1], args.a0, args.a, args.mu, args.ps, args.sc, unper_rank_prob[0])

    table_unper = rf"""\begin{{table}}[h]
\centering
\begin{{threeparttable}}
\centering
\caption{{Price elasticity under unpersonalized ranking}}
\begin{{tabular}}{{lccccccc}}
\hline
& \multicolumn{{3}}{{c}}{{Low-quality}} & & \multicolumn{{3}}{{c}}{{High-quality}} \\
\cline{{2-4}} \cline{{6-8}} 
       & Price  & Ranking   & Demand  &  & Price    & Ranking    & Demand   \\ \hline
Before & {init_per_price1} & {np.around(original_rank_prob[0] * 100, 1)}\%  & {np.around(d1, 3)}   &  & {init_per_price2} & {np.around(100 - original_rank_prob[0] * 100, 1)}\% & {np.around(d2, 3)}    \\
After  & {init_per_price1} & {np.around(unper_rank_prob[0] * 100, 1)}\%  & {np.around(demand_new[0], 3)}  &  & {init_per_price2-0.1}  & {np.around(100 - unper_rank_prob[0] * 100, 1)}\% & {np.around(demand_new[1], 3)}    \\
Change &     &  &  &  &   -0.1 &  +100\%  &  {np.around(demand_new[1] - d2, 3)}  \\
\hline
\end{{tabular}}
\label{{table:price_elasticity_unper}}
\begin{{tablenotes}}
\footnotesize
\item Note: This table shows the impact when the high-quality firm decreases its price by 0.1 and the low-quality firm charges the Q-learning price under unpersonalized ranking.\
``Before'' shows the equilibrium prices charged by Q-learning algorithms and their resulting ranking and demand for low-quality and high-quality firms. \
``After'' shows the price, ranking, and demand when the low-quality firm decreases the price by 0.1. \
``Change'' indicates the difference of price, ranking, and demand after decreasing the price for the high-quality firm.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}"""
    
    with open('./tables/price_elasticity_unper.tex', 'w') as f:
        f.write(table_unper)

    print("Calculating the direct price effect on demand for personalized ranking.")
    rank_prob1 = logistic.cdf(args.a[0] - init_per_price1 - args.a[1] + init_per_price2 - 0.1)
    demand_new = gen_correct_demand([init_per_price1, init_per_price2 - 0.1], args.a0, args.a, args.mu, args.ps)
    print(f"When price increases by 0.1 for high-quality firm:")
    print(f"Probability of low-qaulity firm ranked on the top is {rank_prob1}.")
    print(f"The demand of low-quality firm is {demand_new[0]}.")
    print(f"The demand of high-quality firm is {demand_new[1]}.")
    print(f"\n")

    table_per = rf"""\begin{{table}}[h]
\centering
\begin{{threeparttable}}
\centering
\caption{{Price elasticity under personalized ranking}}
\begin{{tabular}}{{lccccccc}}
\hline
& \multicolumn{{3}}{{c}}{{Low-quality}} & & \multicolumn{{3}}{{c}}{{High-quality}} \\
\cline{{2-4}} \cline{{6-8}}
         & Price  & Ranking   & Demand  &  & Price    & Ranking    & Demand   \\ \hline
Before & {init_per_price1} & {np.around(original_per_rank_prob1 * 100, 1)}\%  & {np.around(original_demand[0], 3)}   &  & {init_per_price2} & {np.around(original_per_rank_prob2 * 100, 1)}\% & {np.around(original_demand[1], 3)}    \\
After  & {init_per_price1} & {np.around(rank_prob1 * 100, 1)}\%  & {np.around(demand_new[0], 3)}  &  & {init_per_price2-0.1}  & {np.around(100 - rank_prob1 * 100, 1)}\% & {np.around(demand_new[1], 3)}    \\
Change &     &  &  &  &   -0.1 &  +{np.around(100 - rank_prob1 * 100 - original_per_rank_prob2 * 100, 1)}\%  &  {np.around(demand_new[1] - original_demand[1], 3)}  \\
\hline
\end{{tabular}}
\label{{table:price_elasticity_per}}
\begin{{tablenotes}}
\footnotesize
\item Note: This table shows the impact when the high-quality firm decreases its price by 0.1 and the low-quality firm charges the Q-learning price under personalized ranking.\
``Before'' shows the equilibrium prices charged by Q-learning algorithms and their resulting ranking and demand for low-quality and high-quality firms. \
``After'' shows the price, ranking, and demand when the low-quality firm decreases the price by 0.1. \
``Change'' indicates the difference of price, ranking, and demand after decreasing the price for the high-quality firm.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}"""
    
    with open('./tables/price_elasticity_per.tex', 'w') as f:
        f.write(table_per)


if __name__ == "__main__":
    args = get_config()
    gen_summary_table(args) # Table 1, 4, 5, 6, 7, 8
    gen_price_elasticity(args) # Table 2, 3