import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from tensorflow.keras.utils import to_categorical
os.environ["CUDA_VISIBLE_DEVICES"]=""


''' return mpe env specified by scenario name '''
def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


''' return sparse action drawn from a distribution specified by logits '''
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


''' policy net architecture '''
class Model(tf.keras.Model):   
    
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation = 'relu')
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.sample_action = ProbabilityDistribution()


    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        hidden_logs1 = self.hidden1(x)
        hidden_logs2 = self.hidden2(hidden_logs1)
        
        return self.logits(hidden_logs2)


    def action_sampler(self, obs):
        logits = self.predict(obs)
        action = self.sample_action.predict(logits)

        return np.squeeze(action, axis=-1)
        
        
class SMPG:

    def __init__(self, models, reward_discount=0.95, cost_discount=1, alpha=1, lr=1e-4):
        self.params = {
            'gamma_reward': reward_discount,
            'gamma_cost': cost_discount,
            'alpha': alpha,
            'lr': lr
        }
        self.models_n = models
        for i in range(env.n):
            self.models_n[i].compile(
                optimizer=ko.Adam(lr=self.params['lr']),
                loss=self.actor_loss
            )
    
    
    def train(self, env, bsize = 48, max_episode = 2000, render = False):
        actions_n = np.zeros((bsize, env.n), dtype = np.int32)
        rewards_n, costs_n, dones_n, values_n = np.empty((4, bsize, env.n))
        # cost history from start to terminal
        term_costs_n = []
        lams_n = np.zeros(env.n)
        # lams history of whole episode
        epi_lams_n = np.zeros((bsize, env.n), dtype = np.float32)
        # same obs space size for each agent
        observations_n = np.zeros((bsize, env.n) + env.observation_space[0].shape, dtype = np.float32)
        rews_history = np.zeros((1,env.n), dtype = np.float32)
        costs_history = np.zeros((1,env.n), dtype = np.float32)
        next_obs_n = np.array(env.reset())
        for episode in range(max_episode):
            for step in range(bsize):
                observations_n[step] = next_obs_n.copy()
                epi_lams_n[step] = lams_n
                for i in range(env.n):
                    actions_n[step, i] = self.models_n[i].action_sampler(next_obs_n[i, :][None, :])
                # same action space size for each agent
                onehot_n = to_categorical(actions_n[step], env.action_space[0].n)
                next_obs_n, rew_con, dones_n[step], _ = [np.array(ret) for ret in env.step(onehot_n)]
                rewards_n[step, :], costs_n[step, :] = rew_con[:, 0], rew_con[:, 1]
                term_costs_n.append(costs_n[step, :])
                rews_history[-1] += rewards_n[step]
                costs_history[-1] += costs_n[step]
                if render:
                    env.render()
                if all(dones_n[step]) or step == (bsize-1):
                    next_obs_n = np.array(env.reset())
                    # update lambda after every terminal state
                    lams_n = self.update_lam(term_costs_n, lams_n)
                    term_costs_n = []
            rews_history = np.append(rews_history, np.zeros((1,env.n),dtype = np.float32), axis = 0)
            costs_history = np.append(costs_history, np.zeros((1,env.n),dtype = np.float32), axis = 0)
            print("Episode: %03d, Agent1 total reward: %03d, Agent 1 total costs: %03d" 
                  % (rews_history.shape[0]-1, rews_history[-2,0], costs_history[-2,0]))
            
            for i in range(env.n):
                weights = self.get_weights(rewards_n[:, i], costs_n[:, i], np.all(dones_n, axis = 1), epi_lams_n[:, i])
                actions_weights = np.concatenate((actions_n[:,i][:, None], weights[:, None]), axis=-1)
                losses = self.models_n[i].train_on_batch(observations_n[:,i,:], actions_weights)
            
        return rews_history[:-1], costs_history[:-1]
    
    
    def get_weights(self, rewards, costs, dones, lams):
        weights = np.zeros(len(rewards)+1)
        for t in reversed(range(rewards.shape[0])):
            weights[t] = rewards[t] - lams[t] * costs[t] + self.params['gamma_reward'] * weights[t+1] * (1-dones[t])
        weights = weights[:-1]
        
        return weights


    def actor_loss(self, actions_weights, logits):
        actions, weights = tf.split(actions_weights, 2, axis=-1)
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=weights)
        
        return policy_loss

    
    def update_lam(self, term_costs_n, lams_n):
        term_costs_n = np.array(term_costs_n)
        discounts = np.power(self.params['gamma_cost'], range(term_costs_n.shape[0]))
        discount_total_costs_n = term_costs_n.T @ discounts
        lams_n += discount_total_costs_n - self.params['alpha']
        lams_n[lams_n < 0] = 0
        
        return lams_n
    

''' plot training result '''  
def cheese_plot(rewards_history, costs_history, save_rate = 1000):
    plt.style.use('seaborn')
    mean_ep_rew = []
    mean_ep_cost = []
    for i in range(len(rewards_history)-save_rate+1):
        mean_ep_rew.append(np.mean(rewards_history[i:save_rate+i]))
        mean_ep_cost.append(np.mean(costs_history[i:save_rate+i]))
    fig = plt.figure()
    plt.plot(np.linspace(save_rate, save_rate+len(mean_ep_rew)-1, num = len(mean_ep_rew), dtype = int),  mean_ep_rew)
    plt.xlabel('Episode')
    plt.ylabel('Mean Episode Reward')
    fig.savefig('Mean_Episode_Rewards_SMPG.png', dpi=fig.dpi)
    fig = plt.figure()
    plt.plot(np.linspace(save_rate, save_rate+len(mean_ep_cost)-1, num=len(mean_ep_cost), dtype = int), mean_ep_cost)
    plt.xlabel('Episode')
    plt.ylabel('Mean Episode Cost')
    fig.savefig('Mean_Episode_Costs_SMPG.png', dpi=fig.dpi)
    plt.show()



if __name__ == '__main__':
    
    bsize = 48
    save_rate = 1000
    epi_max = 5000
    flag = False
    
    env = make_env('constrained_simple_spread_greedy')
    models = [Model(env.action_space[i].n) for i in range(env.n)]
    
    trainer = SMPG(models)
    rewards_history, costs_history = trainer.train(env, bsize = bsize, max_episode = epi_max, render = flag)
    with open('rewards_history_SMPG.pkl', 'wb') as f:
        pickle.dump(rewards_history, f)
    with open('costs_history_SMPG.pkl', 'wb') as f:
        pickle.dump(costs_history, f)
    cheese_plot(rewards_history, costs_history, save_rate)
