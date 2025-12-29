from utils import *
from env import *
import numpy as np
import pickle

def train(seed, comp_prices, mono_prices, args):
    setup_seed(seed)
    num_product = args.num_product

    env, agents = make_env(comp_prices, mono_prices, args)
    all_actions = []
    all_rewards = []
    all_states = []
    num_iter = 0
    if args.learning_algo == 'Q-learning':
        if args.ranking_type == 'unpersonalized2' and args.score_type == 'sales':
            t = 0
            state = env.reset()
            last_best_actions = [np.argmax(agents[i].Q, axis = 1).tolist() for i in range(num_product)]
            all_ranks = []
            while t < args.conv_crit:
                actions = []
                for i in range(num_product):
                    agent = agents[i]
                    action = agent.act(state, num_iter)
                    actions.append(action)
                next_state, reward, rank = env.step_sales(state, actions)
                
                all_ranks.append(rank)
                all_actions.append(actions)
                all_rewards.append(reward)
                all_states.append(state)
                best_actions = [agents[i].update(state, actions[i], reward[i], next_state).tolist() for i in range(num_product)]

                crit = np.array(best_actions) - np.array(last_best_actions)
                crit = np.sum(crit * crit)
                if crit == 0:
                    t += 1
                else:
                    t = 0
                num_iter += 1
                last_best_actions = best_actions
                state = next_state
            
            state_space = env.state_space
            all_states = [all_states, all_ranks]
            return all_rewards, all_actions, [agents[i].Q for i in range(num_product)], all_states, state_space
        else:
            t = 0
            state = env.reset()
            last_best_actions = [np.argmax(agents[i].Q, axis = 1).tolist() for i in range(num_product)]
            while t < args.conv_crit:
                actions = []
                for i in range(num_product):
                    agent = agents[i]
                    action = agent.act(state, num_iter)
                    actions.append(action)
                next_state, reward = env.step(actions)
                
                all_actions.append(actions)
                all_rewards.append(reward)
                all_states.append(state)
                best_actions = [agents[i].update(state, actions[i], reward[i], next_state).tolist() for i in range(num_product)]
                
                crit = np.array(best_actions) - np.array(last_best_actions)
                crit = np.sum(crit * crit)
                if crit == 0:
                    t += 1
                else:
                    t = 0
                num_iter += 1
                last_best_actions = best_actions
                state = next_state
            state_space = env.state_space
            return all_rewards, all_actions, [agents[i].Q for i in range(num_product)], all_states, state_space

    elif args.learning_algo == 'UCB_tuned':
        while num_iter < args.conv_crit:
            actions = []
            for i in range(num_product):
                agent = agents[i]
                action = agent.act()
                actions.append(action)
            state, reward = env.step(actions)
            
            all_actions.append(actions)
            all_rewards.append(reward)
            all_states.append(state)
            for i in range(num_product):
                agents[i].update(actions[i], reward[i])
            num_iter += 1
        state_space = env.state_space
        return all_rewards, all_actions, [agents[i].values for i in range(num_product)], all_states, state_space

def output(seed, comp_prices, mono_prices, args):
    rewards, actions, Qs, states, state_space = train(seed, comp_prices, mono_prices, args)
    results = {'rewards': rewards, 'actions': actions, 'Qs': Qs, 'states': states, 'state_space': state_space, 'args': args}
    output_folder = args.output_dir + args.ranking_type + '/' + args.learning_algo + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if args.file_prefix is not None:
        file_name = f'{args.file_prefix}_n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}_seed{seed}.pkl'
    else:
        file_name = f'n{args.num_product}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}_\
alpha{args.alpha}_beta{args.beta}_delta{args.delta}_seed{seed}.pkl'
    with open(output_folder + file_name, 'wb') as f:
        pickle.dump(results, f)
