import numpy as np
import copy
import itertools
import os
import sys
sys.path.insert(1, os.path.realpath(os.path.pardir))
from utils import *
import pickle

class Agent():
    def __init__(self, action_space, state_space, alpha, beta, delta):
        self.action_space = action_space
        self.action_size = len(self.action_space)
        self.state_space = state_space
        self.state_size = len(self.state_space)
        self.beta = beta # exploration rate
        self.alpha = alpha # learning rate
        self.delta = delta # discount factor
        self.Q = np.zeros((self.state_size, self.action_size))
        
    def act(self, state, num_iter):
        self.epsilon = np.exp(-self.beta * num_iter)
        crit = np.random.uniform(0, 1, 1)
        if crit < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action_idx = np.argmax(self.Q[self.state_space.index(state)])
            action = self.action_space[action_idx]
        return action
    
    def update(self, state, action, reward, next_state):
        state_idx = self.state_space.index(state)
        next_state_idx = self.state_space.index(next_state)
        action_idx = self.action_space.index(action)
        old_value = self.Q[state_idx, action_idx]
        next_max = np.max(self.Q[next_state_idx])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.delta * next_max)
        self.Q[state_idx, action_idx] = new_value
        best_action = np.argmax(self.Q, axis = 1)
        return best_action

class UCB_tuned_Agent():
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_size = len(self.action_space)
        self.counts = np.zeros(self.action_size)
        self.total_profit_square = np.zeros(self.action_size)
        self.mean_profit = np.zeros(self.action_size)
        self.t = 0
    
    def permute(self):
        return np.random.permutation(self.action_space)

    def act(self):
        self.t += 1
        ucb_index = np.zeros(self.action_size)
        V = np.log(2 * self.t/self.counts) + self.total_profit_square - self.mean_profit**2
        ucb_index = self.mean_profit + np.sqrt(np.log(self.t) / self.counts * np.where(V < 0.25, V, 0.25))
        action = self.action_space[np.argmax(ucb_index)]
        return action

    def update(self, action, reward):
        action_idx = self.action_space.index(action)
        self.counts[action_idx] += 1
        self.total_profit_square[action_idx] += (reward - self.mean_profit[action_idx])**2 / self.counts[action_idx]
        self.mean_profit[action_idx] += (reward - self.mean_profit[action_idx]) / self.counts[action_idx]

def init_UCB(env, agents):
    action_spaces = [agent.permute() for agent in agents]
    num_products = len(agents)
    for i in range(len(action_spaces[0])):
        actions = [action_spaces[j][i] for j in range(num_products)]
        state, rewards = env.step(actions)
        for j in range(num_products):
            agents[j].update(actions[j], rewards[j])
    return agents

class Environment():
    def __init__(self, state_space, reward_space, args = None):
        self.state_space = state_space
        self.reward_space = reward_space
        self.state_size = len(self.state_space)
        self.args = args
    
    def reset(self):
        state_idx = np.random.choice(self.state_size)
        return self.state_space[state_idx]
    
    def step(self, actions):
        self.state = actions
        action_index = self.state_space.index(actions)
        rewards = self.reward_space[action_index]
        return self.state, rewards
    
    def step_sales(self, state, actions):
        rank, _ = gen_sales_ranking(state, self.args.kappa)
        demand = gen_demand_unper_two_product(actions, self.args.a0, self.args.a, self.args.mu, self.args.ps, self.args.sc, rank)
        rewards = gen_profit_unper_two_product(actions, self.args.a0, self.args.a, self.args.mc, self.args.mu, self.args.ps, self.args.sc, rank)
        self.state = list(demand)
        return self.state, rewards, rank

def init_Q(env, agents, ind_sales = False, args = None):
    num_product = len(agents)
    state_space = np.array(env.state_space)
    
    if ind_sales:
        for i in range(num_product):
            agent = agents[i]
            Q = agent.Q
            action_space = agent.action_space
            for j in range(agent.action_size):
                competitor_agent = agents[1 - i]
                competitor_action = competitor_agent.action_space
                action = action_space[j]
                all_reward = 0.
                for k in range(len(competitor_action)):
                    price = [action, competitor_action[k]]
                    rank, rank_prob = gen_sales_ranking(price, args.kappa)
                    profit1 = gen_profit_unper_two_product(price, args.a0, args.a, args.mc, args.mu, args.ps, args.sc, 1)
                    profit2 = gen_profit_unper_two_product(price, args.a0, args.a, args.mc, args.mu, args.ps, args.sc, 0)
                    expected_profit1 = rank_prob[0] * np.array(profit1)
                    expected_profit2 = rank_prob[1] * np.array(profit2)
                    expected_profit = expected_profit1 + expected_profit2
                    all_reward += expected_profit[i]

                Q[:,j] = all_reward/len(competitor_action)
            agent.Q = Q/(1 - agent.delta)
    else:
        for i in range(num_product):
            agent = agents[i]
            Q = agent.Q
            action_space = agent.action_space
            
            for j in range(agent.action_size):
                actions = copy.deepcopy(state_space)
                actions[:,i] = action_space[j]
                actions = np.unique(actions, axis=0)
                all_reward = 0.
                
                for k in range(len(actions)):
                    _, rewards = env.step(actions[k].tolist())
                    all_reward += rewards[i]
                    
                Q[:,j] = all_reward/len(actions)
            agent.Q = Q/(1 - agent.delta)
    return agents

def make_env(comp_prices, mono_prices, args):
    num_product = args.num_product
    if num_product == 2 or args.ranking_type == "personalized3":
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
            elif args.learning_algo == "UCB_tuned":
                action_space = np.linspace(start_price, end_price, args.n_action)
            action_space = np.around(action_space, 3).tolist()
            action_spaces.append(action_space)
        state_space = list(list(tup) for tup in itertools.product(*action_spaces))
        
        reward_space = []
        if args.ranking_type == "personalized" or args.ranking_type == "personalized3":
            if args.score_type == "risk-averse":
                file_name = f"reward_space_{args.ranking_type}_{args.file_prefix}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}\
_{args.num_customer}_{args.n_sim}_{args.r}.pkl"
                with open(args.data_dir + file_name, 'rb') as f:
                    results = pickle.load(f)
                state_space = results['state_space']
                reward_space = results['reward_space']
                print(f"Finish loading reward space from {file_name}.")
            else:
                for i in range(len(state_space)):
                    rewards = gen_per_profit_correct(state_space[i], args.a0, args.a, args.mu, args.ps, args.mc)
                    reward_space.append(rewards.tolist())
        elif args.ranking_type == "unpersonalized2":
            if args.score_type == "sales":
                demand_spaces1 = []
                demand_spaces0 = []
                for i in range(len(state_space)):
                    rewards = gen_profit_unper_two_product(state_space[i], args.a0, args.a, args.mc, args.mu, args.ps, args.sc, 1)
                    reward_space.append(rewards)
                    rewards = gen_profit_unper_two_product(state_space[i], args.a0, args.a, args.mc, args.mu, args.ps, args.sc, 0)
                    reward_space.append(rewards)
                    demand = gen_demand_unper_two_product(state_space[i], args.a0, args.a, args.mu, args.ps, args.sc, 1)
                    demand_spaces1.append(list(demand))
                    demand = gen_demand_unper_two_product(state_space[i], args.a0, args.a, args.mu, args.ps, args.sc, 0)
                    demand_spaces0.append(list(demand))
                state_space = demand_spaces1 + demand_spaces0
            elif args.score_type == "risk-averse":
                file_name = f"reward_space_{args.ranking_type}_{args.file_prefix}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}\
_{args.num_customer}_{args.n_sim}_{args.r}.pkl"
                with open(args.data_dir + file_name, 'rb') as f:
                    results = pickle.load(f)
                state_space = results['state_space']
                reward_space = results['reward_space']
                print(f"Finish loading reward space from {file_name}.")
            else:
                for i in range(len(state_space)):
                    _, rewards = gen_expected_profit_unper_two_product(state_space[i], args)
                    reward_space.append(rewards)
        elif args.ranking_type == "imperfect":
            for i in range(len(state_space)):
                rewards = gen_profit_per_inaccurate_rank(state_space[i], args.a0, args.a, args.mc, args.lamb, args.mu, args.sc, args.ps)
                reward_space.append(rewards)
        elif args.ranking_type == "imperfect-rational" or args.ranking_type == "imperfect-rational-n_customer5000":
            file_name = f"reward_space_{args.ranking_type}_{args.file_prefix}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}\
_{args.num_customer}_{args.n_sim}_{args.r}.pkl"
            with open(args.data_dir + file_name, 'rb') as f:
                results = pickle.load(f)
            state_space = results['state_space']
            reward_space = results['reward_space']
            
    elif num_product == 3 and args.ranking_type == "unpersonalized3":
        reward_space_file_name = f"reward_space_{args.ranking_type}_a0{args.a0}_a{args.a}_mc{args.mc}_gamma{args.gamma}_sc{args.sc}\
_{args.num_customer}_{args.n_sim}_{args.step_size_init}.pkl"
        with open(args.data_dir + reward_space_file_name, 'rb') as f:
            results = pickle.load(f)
        state_space = results['state_space']
        reward_space = results['reward_space']
        action_spaces = results['action_spaces']
    
    env = Environment(state_space, reward_space, args)
    if args.learning_algo == "Q-learning":
        if args.state == "all_prices":
            agents = [Agent(action_spaces[i], state_space, args.alpha, args.beta, args.delta) for i in range(num_product)]
            agents = init_Q(env, agents)
        elif args.state == "sales":
            agents = [Agent(action_spaces[i], state_space, args.alpha, args.beta, args.delta) for i in range(num_product)]
            agents = init_Q(env, agents, ind_sales = True, args = args)
    elif args.learning_algo == "UCB_tuned":
        agents = [UCB_tuned_Agent(action_spaces[i]) for i in range(num_product)]
        agents = init_UCB(env, agents)

    return env, agents