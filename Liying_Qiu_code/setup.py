import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_product', type=int, default=2,
                        help='number of products')
    parser.add_argument('--a0', type=float, default=0.0,
                        help='outside option value')
    parser.add_argument('--a', type=float, nargs='+', default=[4., 4.5],
                        help='vertical differentiation')
    parser.add_argument('--mc', type=float, nargs='+', default=[1., 1.5],
                        help='marginal cost')
    parser.add_argument('--mu', type=float, default=1,
                        help='horizontal differentiation')
    parser.add_argument('--ps', type=float, default=1,
                        help='price sensitivity (positive value)')
    parser.add_argument('--sc', type=float, default=1.5,
                        help='search cost')
    parser.add_argument('--gamma', type=float, default=-1.,
                        help='price effect under unpersonalized ranking')
    parser.add_argument('--lamb', type=float, default=1.0,
                        help='accuracy under imperfect personalized ranking')
    parser.add_argument('--xi', type=float, default=0.1,
                        help='action space parameter')
    parser.add_argument('--r', type=float, default=0,
                        help='risk coefficient, indicates risk aversion if r > 0')
    parser.add_argument('--step_size', type=float, default=0.25,
                        help='action space interval size')
    parser.add_argument('--step_size_init', type=float, default=0.2,
                        help='initial action space interval size to calculate the price range')
    parser.add_argument('--n_action', type=int, default=10,
                        help='number of prices in the action space when using UCB')
    parser.add_argument('--num_customer', type=int, default=10000,
                        help='number of customers')
    parser.add_argument('--n_sim', type=int, default=1000,
                        help='number of simulations')
    parser.add_argument('--n_mpe', type=int, default=30,
                        help='number of successful simulations for the MPE')
    parser.add_argument('--kappa', type=float, default=1,
                        help='weight for the error term in the ranking score')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--beta', type=float, default=2e-6,
                        help='exploration rate for epsilon-greedy')
    parser.add_argument('--delta', type=float, default=0.95,
                        help='discount factor')
    parser.add_argument('--conv_crit', type=int, default=100000,
                        help='convergence criterion')
    parser.add_argument('--ranking_type', type=str, default='unpersonalized2', 
                        choices=['unpersonalized2', 
                                 'personalized', 
                                 'imperfect', 
                                 'unpersonalized3', 
                                 'personalized3',
                                 'imperfect-rational',
                                 'imperfect-rational-n_customer5000'],
                        help='ranking type')
    parser.add_argument('--score_type', type=str, default='expected-utility', 
                        choices=['expected-utility', 
                                 'sales', 
                                 'risk-averse'],
                        help='ranking score type under unpersonalized ranking')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='starting seed')
    parser.add_argument('--end_seed', type=int, default=100,
                        help='ending seed')
    parser.add_argument('--n_sum', type=int, default=1000,
                        help='number of samples for summarization')
    parser.add_argument('--n_steps', type=int, default=30,
                        help='number of steps for simulating MPE profit.')
    parser.add_argument('--output_dir', type=str, default='./results/',
                        help='output directory')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='data directory')
    parser.add_argument('--figure_dir', type=str, default='./figures/',
                        help='figure directory')
    parser.add_argument('--learning_algo', type=str, default='Q-learning', 
                        choices=['Q-learning', 
                                 'UCB_tuned'],
                        help='reinforcement learning algorithm')
    parser.add_argument('--state', type=str, default='all_prices', 
                        choices=['all_prices', 'sales'],
                        help='state representation')
    parser.add_argument('--n_cores', type=int, default=96,
                        help='number of cores')
    parser.add_argument('--do_summary', action='store_true',
                        help='Whether to summarize the results')
    parser.add_argument('--do_rewrite', action='store_true',
                        help='Whether to rewrite everything')
    parser.add_argument('--do_rewrite_cs', action='store_true',
                        help='Whether to recalculate consumer surplus')
    parser.add_argument('--do_rewrite_profit_gain', action='store_true', 
                        help='Whether to recalculate profit gain')
    parser.add_argument('--do_rational_belief', action='store_true',
                        help='Whether to use rational belief for consumers')
    parser.add_argument('--file_prefix', default=None, type=str,
                        help='File name prefix for the output files')
    parser.add_argument('--table_label', default=None, type=str,
                        help='Table label for the output files')
    parser.add_argument('--table_name', default=None, type=str,
                        help='Table name for the output files')
    parser.add_argument('--table_caption', default=None, type=str,
                        help='Table caption for the output files')
    parser.set_defaults(do_summary=True)
    args = parser.parse_args()
    assert len(args.a) == len(args.mc), "The length of variable a and mc should be the same."
    assert len(args.a) == args.num_product, f"The length of a should be the same as the number of products. \
Now the number of products is {args.num_product}. The length of a is {len(args.a)}."
    return args