import multiprocessing as mp
from train import *
from utils import *
from summarize import *
from figure import *
from setup import get_config

def main(args):
    if args.file_prefix is not None and args.file_prefix != 'UCB':
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
        
    if os.path.exists(args.data_dir + file_name) and not args.do_rewrite:
        with open(args.data_dir + file_name, 'rb') as f:
            data = pickle.load(f)
        comp_prices = data['comp_prices']
        mono_prices = data['mono_prices']
        comp_profit = data['comp_profit']
        mono_profit = data['mono_profit']
        print(f"File already existed. Loading existing price range for {args.ranking_type} ranking.\n")
    else:
        if args.ranking_type == "unpersonalized2":
            if args.score_type == 'sales':
                comp_prices, mono_prices = gen_price_range_mpe(args, seed = 0)
                mono_profit = None
                comp_profit = None
            elif args.score_type == 'risk-averse':
                assert args.r > 0 
                comp_prices, mono_prices = gen_price_range_unpersonalized2_risk_utility(args)
                mono_profit = None
                comp_profit = None
                gen_reward_space_unpersonazlied2_risk_utility(comp_prices, mono_prices, args)
            else:
                comp_prices, mono_prices, comp_profit, mono_profit = gen_price_range_unpersonalized2(args.a0, args.a, args.mc, args.mu, args.ps, args.gamma, args.sc, 
                                                                                                     step_size = 0.01, score_type = args.score_type, args = args)
        elif args.ranking_type == "personalized" or args.ranking_type == "personalized3":
            if args.score_type == 'risk-averse':
                assert args.r > 0
                comp_prices, mono_prices = gen_price_range_personalized2_risk_utility(args)
                mono_profit = None
                comp_profit = None
                gen_reward_space_personalized2_risk_utility(comp_prices, mono_prices, args)
            else:
                comp_prices, mono_prices, comp_profit, mono_profit = gen_per_price_range(args.a0, args.a, args.mc, args.ps, args.mu)
        elif args.ranking_type == "unpersonalized3":
            comp_prices, mono_prices, comp_profit, mono_profit = gen_price_range_unpersonalized3(args)
        elif args.ranking_type == "imperfect-rational" or args.ranking_type == "imperfect-rational-n_customer5000":
            comp_prices, mono_prices = gen_price_range_imperfect_personalized(args)
            mono_profit = None
            comp_profit = None
            gen_reward_space_imperfect_per_rational(comp_prices, mono_prices, args)

        if args.do_rewrite:
            print(f"File already existed. Rewriting price range for {args.ranking_type} ranking.\n")
        else:
            print(f"File not created before. Finishing generating price range for {args.ranking_type} ranking.\n")
        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir)
        with open(args.data_dir + file_name, 'wb') as f:
            pickle.dump({'comp_prices': comp_prices, 'mono_prices': mono_prices, 'comp_profit': comp_profit, 'mono_profit': mono_profit}, f)
    
    print(f"Competitive prices: {comp_prices}")
    print(f"Monopoly prices: {mono_prices}")
    print(f"Competitive profits: {comp_profit}")
    print(f"Monopoly profits: {mono_profit}")

    if args.ranking_type == "unpersonalized3":
        gen_action_reward_space_unpersonalized3(comp_prices, mono_prices, args)

    output_folder = args.output_dir + args.ranking_type + '/' + args.learning_algo + '/'
    if args.file_prefix is not None:
        file_name_prefix = f'{args.file_prefix}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}_seed'
    else:
        file_name_prefix = f'n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}_seed'
    
    if not args.do_rewrite:
        try:
            existing_files = [f for f in os.listdir(output_folder) if f.startswith(file_name_prefix)]
            existing_seeds = [int(f.split('seed')[1].split('.')[0]) for f in existing_files]
            if len(existing_seeds) > 0:
                print(f"Existing seeds: {existing_seeds}")
            if max(existing_seeds) + 1 < args.end_seed:
                print(f"Training is not finished. Restart training from seed {max(existing_seeds) + 1}.")
                start_seed = max(existing_seeds)
            elif max(existing_seeds) + 1 >= args.end_seed:
                print(f"Training is finished.")
                start_seed = args.end_seed
        except:
            start_seed = args.start_seed
    else:
        print(f"Rewriting everything.")
        start_seed = args.start_seed

    if start_seed < args.end_seed:
        print(f"Start training from seed {start_seed}.")
        print(f"Number of cores: {mp.cpu_count()}")
        pool = mp.Pool(mp.cpu_count())
        processes = [pool.apply_async(output, args=(seed, comp_prices, mono_prices, args)) for seed in range(start_seed, args.end_seed)]
        results = [p.get() for p in processes]
        pool.close()
    else:
        print(f"Training is already finished.")

    if args.do_summary:
        output_profit_gain(args)
        output_consumer_surplus(args)
    
if __name__ == "__main__":
    args = get_config()
    main(args)