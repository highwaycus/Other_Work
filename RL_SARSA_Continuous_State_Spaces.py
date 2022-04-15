'''
Environments (Open AI gym, https://gym.openai.com/):  Frozen Lake 8x8, Cart Pole, Lunar Lander (Box2D package)
Implementing SARSA algorithm on these three environments while representing their state spaces as a continuous vector.

Frozen Lake 8x8: https://gym.openai.com/envs/FrozenLake8x8-v0/
Cart Pole: https://gym.openai.com/envs/CartPole-v1/
Lunar Lander: https://gym.openai.com/envs/LunarLanderContinuous-v2/
'''

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import math
from initial_startup import prepFrozen8, prepCartPole, prepLunarLender


def decay_factor(numepisodes, lowepsilon, highepsilon):
    ratio = np.log(lowepsilon / highepsilon)
    ans = np.exp( ratio / (0.7 * numepisodes))
    return ans


#################################################
def assign_env(env_name):
    assert env_name in ['Frozen8', 'CartPole', 'Lunar_Lander']
    print('Environment: {}'.format(env_name))
    if env_name == 'Frozen8':
        env, nS, nA, phi, dname = prepFrozen8()
        total_epi = 100000
        # Q = np.ones((nS, nA))
    elif env_name == 'CartPole':
        env, nS, nA, phi, dname = prepCartPole()
        total_epi = 100000
        # Q = np.random.randn(10, 10, 10, 10, env.action_space.n)
        # Q = np.zeros((10, 10, 10, 10) + (env.action_space.n,))
    else:
        env, nS, nA, phi, dname = prepLunarLander()
        total_epi = 30000
        # Q = []
    W = {j: np.mat([[np.random.rand(1).item()] for i in range(nS)]) for j in range(nA)}
    # W = {j: np.mat([[1] for i in range(nS)]) for j in range(nA)}
    return env, nS, nA, phi, total_epi, W


def discretised_state(state, env):
    if env.spec._env_name == 'CartPole':
        # Discretize space to 10^4 states
        discrete_state_size = 10
        discrete_state = np.array([0, 0, 0, 0])
        for i in range(4):
            space_interval = (env.observation_space.high[i] - env.observation_space.low[i]) / discrete_state_size
            discrete_state[i] = (state[i] - env.observation_space.low[i]) // space_interval
            discrete_state[i] = min(discrete_state_size - 1, max(0, discrete_state[i]))
        return tuple(discrete_state.astype(np.int))
    else:
        return state


def sufficient_exploration_early_check(env_name, r_):
    if env_name == 'Frozen8':
        if r_ >= 0.6:
            return True
    elif env_name == 'CartPole':
        if r_ > 300:
            return True
    else:
        # positive reward on average on Lunar Lander
        if r_ > 0:
            return True

    return False


def execute_policy(W, env, phi, n_episode=500, max_step=1000, regularize=False):
    """
    : return: float, quality
    """
    reward = 0
    step_record = []
    for epi in range(n_episode):
        done = False
        np.random.seed()
        state = env.reset()
        i = 0
        while (not done) and (i < max_step):
            action = greedy_func(W=W, env=env, state=state, epsilon=-1, phi=phi, regularize=regularize)
            state, r, done, _ = env.step(action)
            reward += r
            if done:
                step_record.append(i)
            i += 1
    return reward / n_episode


def greedy_func(W, env, state, epsilon, phi, regularize):
    if np.random.uniform(0, 1) < epsilon:
        action_next = [env.action_space.sample()]
    else:
        max_v = -np.Inf
        action_next = []
        state_phi = phi(state)
        for a in range(len(W)):
            Wa_a = W[a]
            # if type(state) is int:
            #     state = [int(i == state) for i in range(len(Wa_a))]
            tmp = np.dot(np.mat(state_phi), Wa_a).item()
            if regularize:
                lambda_a = 0.1
                regu_term = sum([(Wa_a[j]) ** 2 for j in range(len(Wa_a))])
                tmp = tmp - lambda_a * regu_term
            # I want to randomly choose action if the outcome ties
            if tmp > max_v:
                max_v = tmp
                action_next = [a]
            elif tmp == max_v:
                action_next.append(a)
    return random.choice(action_next)


def sarsa_main(env_name, alpha=0.01, gamma=1, epsilon=0.5, episode_min=2000, regularize=False, best_para=False, episilon_constant=False):
    # Generate Env
    env, nS, nA, phi, total_epi_n, W_a = assign_env(env_name)

    # Initialize
    r_list = []
    episode = 0
    policy_test = {}
    decay_rate = decay_factor(total_epi_n, lowepsilon=0.01, highepsilon=0.5)
    if episilon_constant:
        decay_rate = 1
    mean_quality = 0
    best_para_ = W_a
    previous_quality = -np.Inf
    while (episode < episode_min) or (not sufficient_exploration_early_check(env_name,  mean_quality)):
        # print(W_a[0].transpose())
        # init
        t = 0

        state_t = env.reset()
        a_t = greedy_func(W_a, env, state_t, epsilon, phi, regularize)
        # In 1 episode, for each step
        r = 0
        while t < env._max_episode_steps:
            state_t_1, reward, done, info = env.step(a_t)
            a_t_1 = greedy_func(W_a, env, state_t_1, epsilon, phi, regularize)
            phi_state_t = np.mat(phi(state_t)).transpose()
            Q_sa_state_t = np.dot(np.mat(phi(state_t)), W_a[a_t]).item()
            Q_sa_state1_t1 = np.dot(np.mat(phi(state_t_1)), W_a[a_t_1]).item()
            if done:
                W_a[a_t] = W_a[a_t] + alpha * (reward - Q_sa_state_t) * phi_state_t
            else:
                W_a[a_t] = W_a[a_t] + alpha * (
                            reward + gamma * Q_sa_state1_t1 - Q_sa_state_t) * phi_state_t
            state_t = state_t_1
            a_t = a_t_1
            t += 1
            r += reward
            if done:
                break
            # else:
            #     env.render()
        r_list.append(r)
        episode += 1
        if best_para and (episode % best_para == 0) and (episode > 5):
            if previous_quality > np.mean(r_list[-5:]):
                W_a = best_para_
            else:
                best_para_ = W_a
                previous_quality = np.mean(r_list[-5:])
        if epsilon > 0.01:
            epsilon *= decay_rate
        if episode % episode_min == 0:
            policy_test[episode] = {}
            for rt in range(5):
                quality1000 = execute_policy(W_a, env, phi, n_episode=200, max_step=env._max_episode_steps, regularize=regularize)
                policy_test[episode][rt] = {'quality': quality1000, 'epsilon': epsilon}
            mean_quality = np.mean([policy_test[episode][rt_i]['quality'] for rt_i in policy_test[episode]])
            print('episode={}'.format(episode), 'quality={}'.format(mean_quality))
            if sufficient_exploration_early_check(env_name, mean_quality):
                break
    print('N of Episodes: {}'.format(episode + 1))
    env.close()
    return policy_test


def plot_task_1(use_record, env_name, plot_file=''):
    try:
        os.makedirs(plot_file)
    except:
        pass
    mean_iter = [np.mean([use_record[epi_][r]['quality'] for r in use_record[epi_]]) for epi_ in use_record]
    std_iter = [np.std([use_record[epi_][r]['quality'] for r in use_record[epi_]]) for epi_ in use_record]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.errorbar([epi_ for epi_ in use_record], mean_iter, yerr=std_iter)
    ax2.plot([epi_ for epi_ in use_record],[use_record[epi_][0]['epsilon'] for epi_ in use_record], 'g-')
    plt.title(env_name)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel(eval_item.capitalize())
    ax2.set_ylabel('Epsilon value')
    plt.savefig(plot_file + 'RL_hw3_task1-' + env_name + '.jpg')


#############################################################
def main(main_dir):
    episode_min = 2000
    alpha = 0.01
    gamma = 1
    epsilon = 0.5
    #############################
    # FrozenLake8
    print('###################################')
    env_name = 'Frozen8'
    policy_test =sarsa_main(env_name, alpha=alpha, gamma=gamma, epsilon=epsilon, episode_min=episode_min)
    print('Done')
    plot_task_1(policy_test, env_name, plot_file=main_dir + 'plot/')
    #############################
    # CartPole
    print('###################################')
    env_name = 'CartPole'
    policy_test = sarsa_main(env_name, alpha=alpha, gamma=gamma, epsilon=epsilon, episode_min=episode_min, regularize=False, best_para=False, episilon_constant=True)
    print('Done')
    plot_task_1(policy_test, env_name, plot_file=main_dir + 'plot/')
    #############################
    # Lunar
    print('###################################')
    env_name = 'Lunar_Lander'
    policy_test = sarsa_main(env_name, alpha=alpha, gamma=gamma, epsilon=epsilon, episode_min=episode_min)
    print('Done')
    plot_task_1(policy_test, env_name, plot_file=main_dir + 'plot/')


########################################
########################################
if __name__ == '__main__':
    main_dir = ''
    main(main_dir=main_dir)
