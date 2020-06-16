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


''' policy network architecture '''
class p_model(tf.keras.Model):   
    
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
        

''' value network architecture '''
class v_model(tf.keras.Model):
    
    def __init__(self):
        super().__init__('mlp_value')
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation = 'relu')
        self.values = kl.Dense(1, name='critic_value')


    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        hidden_logs1 = self.hidden1(x)
        hidden_logs2 = self.hidden2(hidden_logs1)
        
        return self.values(hidden_logs2)

    
class SMA2C:

    def __init__(self, p_models, v_models):
        self.params = {
            'gamma_reward': 0.95,
            'gamma_cost':1,
            'alpha':1
        }
        self.p_models_n = p_models
        self.v_models_n = v_models
        for i in range(env.n):
            self.p_models_n[i].compile(
                optimizer=ko.Adam(lr=0.0005),
                loss = self.actor_loss
            )
            self.v_models_n[i].compile(
                optimizer=ko.Adam(lr=0.0005),
                loss = kls.mean_squared_error
            )
    
    
    def train(self, env, bsize = 16, max_episode = 2000, reset_rate = 5, render = False):
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
            # reset env every reset_rate episodes for exploration
            if (episode+1) % reset_rate == 0:
                next_obs_n = np.array(env.reset())
                # update lambda after every terminal state
                if not len(term_costs_n) == 0:
                    lams_n = self.update_lam(term_costs_n, lams_n)
                    term_costs_n = []
            for step in range(bsize):
                observations_n[step] = next_obs_n.copy()
                epi_lams_n[step, :] = lams_n
                for i in range(env.n):
                    actions_n[step, i] = self.p_models_n[i].action_sampler(next_obs_n[i, :][None, :])
                    values_n[step, i] = self.v_models_n[i].predict(next_obs_n.flatten()[None, :])
                # same action space size for each agent
                onehot_n = to_categorical(actions_n[step], env.action_space[0].n)
                next_obs_n, rew_con, dones_n[step], _ = [np.array(ret) for ret in env.step(onehot_n)]
                rewards_n[step, :], costs_n[step, :] = rew_con[:, 0], rew_con[:, 1]
                term_costs_n.append(costs_n[step, :])
                if render:
                    env.render()
                rews_history[-1] += rewards_n[step]
                costs_history[-1] += costs_n[step]
                if all(dones_n[step]):
                    next_obs_n = np.array(env.reset())
                    # update lambda after every terminal state
                    lams_n = self.update_lam(term_costs_n, lams_n)
                    term_costs_n = []
            rews_history = np.append(rews_history, np.zeros((1,env.n),dtype = np.float32), axis = 0)
            costs_history = np.append(costs_history, np.zeros((1,env.n),dtype = np.float32), axis = 0)
            print("Episode: %03d, Agent1 total reward: %03d, Agent 1 total costs: %03d" 
                  % (rews_history.shape[0]-1, rews_history[-2,0], costs_history[-2,0]))
            
            for i in range(env.n):
                next_value = self.v_models_n[i].predict(next_obs_n.flatten()[None, :])
                targets, advs = self.targets_advantages(rewards_n[:,i], np.all(dones_n, axis = 1), 
                                                         values_n[:,i], next_value, costs_n[:,i], epi_lams_n[:,i])
                acts_and_advs = np.concatenate((actions_n[:,i][:, None], advs[:, None]), axis=-1)
                p_losses = self.p_models_n[i].train_on_batch(observations_n[:,i,:], acts_and_advs)
                v_losses = self.v_models_n[i].train_on_batch(np.reshape(observations_n, 
                                          (observations_n.shape[0], observations_n.shape[1]*observations_n.shape[2])), targets)
        # save policy models
        for i in range(env.n):
            self.p_models_n[i].save_weights('pmodel%d' %i)
#            self.p_models_n[i].save('pmodel%d' %i)
            
        return rews_history[:-1], costs_history[:-1]

    
    ''' return value targets and estimated advantages '''
    def targets_advantages(self, rewards, dones, values, next_value, costs, lams):
        next_values = np.append(values, next_value)
        targets = np.zeros(len(rewards))
        for t in reversed(range(rewards.shape[0])):
            targets[t] = rewards[t] - lams[t] * costs[t] + self.params['gamma_reward'] * next_values[t+1] * (1-dones[t])
        advantages = targets - values
        
        return targets, advantages

    
    ''' loss for actor net '''
    def actor_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        
        return policy_loss

    
    ''' return updated lagrange multiplier for each agent i '''
    def update_lam(self, term_costs_n, lams_n):
        term_costs_n = np.array(term_costs_n)
        discounts = np.power(self.params['gamma_cost'], range(term_costs_n.shape[0]))
        discount_total_costs_n = term_costs_n.T @ discounts
        lams_n += discount_total_costs_n - self.params['alpha']
        lams_n[lams_n < 0] = 0
        
        return lams_n
    
    
    ''' test policy model in env '''
    def test(self, env, maxstep = 100, render = True, report_rate = 5):
        actions_n = np.zeros(env.n, dtype = np.int32)
        rews_history = np.zeros((1,env.n), dtype = np.float32)
        costs_history = np.zeros((1,env.n), dtype = np.float32)
        next_obs_n = np.array(env.reset())
        for t in range(maxstep):
            for i in range(env.n):
                actions_n[i] = self.p_models_n[i].action_sampler(next_obs_n[i, :][None, :])
            onehot_n = to_categorical(actions_n, env.action_space[0].n)
            next_obs_n, rew_con, dones_n, _ = [np.array(ret) for ret in env.step(onehot_n)]
            rewards_n, costs_n = rew_con[:, 0], rew_con[:, 1]
            if render:
                env.render()
            rews_history[-1] += rewards_n
            costs_history[-1] += costs_n
            if all(dones_n):
                next_obs_n = np.array(env.reset())
            rews_history = np.append(rews_history, np.zeros((1,env.n),dtype = np.float32), axis = 0)
            costs_history = np.append(costs_history, np.zeros((1,env.n),dtype = np.float32), axis = 0)
            if t%report_rate == 0:
                print("Test step: %03d, Average reward: %03d, Average costs: %03d" 
                  % (rews_history.shape[0]-1, np.mean(rews_history[-2]), np.mean(costs_history[-2])))
            
        return rews_history[:-1], costs_history[:-1]   


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
    fig.savefig('Mean_Episode_Rewards_SMA2C.png', dpi=fig.dpi)
    fig = plt.figure()
    plt.plot(np.linspace(save_rate, save_rate+len(mean_ep_cost)-1, num=len(mean_ep_cost), dtype = int), mean_ep_cost)
    plt.xlabel('Episode')
    plt.ylabel('Mean Episode Cost')
    fig.savefig('Mean_Episode_Costs_SMA2C.png', dpi=fig.dpi)
    plt.show()


if __name__ == '__main__':

    save_rate = 1000
    batch_size = 16
    epi_max = 8000
    reset_rate = 5
    ifrender = True
    
    env = make_env('constrained_simple_spread_greedy')
    p_models = [p_model(env.action_space[i].n) for i in range(env.n)]
    v_models = [v_model() for i in range(env.n)]
    
    trainer = SMA2C(p_models, v_models)
    rewards_history, costs_history = trainer.train(env, bsize = batch_size, max_episode = epi_max, 
                                                       reset_rate = reset_rate, render = ifrender)
    with open('rewards_history_SMA2C.pkl', 'wb') as f:
        pickle.dump(rewards_history, f)
    with open('costs_history_SMA2C.pkl', 'wb') as f:
        pickle.dump(costs_history, f)
    cheese_plot(rewards_history, costs_history, save_rate)
