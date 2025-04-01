import os
import time

os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '64'
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '64'
import sys


from feature_env import FeatureEnv, SUPPORT_STATE_METHOD, REPLAY
from initial import init_param
from model import operation_set, ClusterActorCritic, OpActorCritic, O1, O2
from replay import RandomClusterReplay, RandomOperationReplay
from replay import Replay, PERClusterReplay, PEROperationReplay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
base_path = './data/processed'
import warnings
import torch
import pandas as pd
import numpy as np
import math

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

torch.set_num_threads(32)
from utils.logger import *

info(torch.get_num_threads())
info(torch.__config__.parallel_info())
import warnings

torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

import json
import pickle

"""
This script is for a more robust RL agent, if one potential error occur, the process will retreat to a safe state.
"""

def train(param):
    if param['cuda'] == 'cpu':
        cuda_info = None
        info(f'running experiment on cpu')
    else:
        cuda_info = True
        info(f'running experiment on cuda')
    start_time = str(time.asctime())
    feature_path = './logfile/feature/' + param['name'] + '/'
    performance_path = './logfile/performance/' + param['name'] + '/'
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    if not os.path.exists(performance_path):
        os.mkdir(performance_path)

    os.mkdir(feature_path + start_time + '/')
    feature_path = feature_path + start_time + '/'
    os.mkdir(performance_path + start_time + '/')
    performance_path = performance_path + start_time + '/'    
    
    STATE_METHOD = param['state_method']

    MODE = param['train_mode']
    assert STATE_METHOD in SUPPORT_STATE_METHOD
    NAME = param['name']
    DISTANCE = param['distance']
    ENV = FeatureEnv(task_name=NAME, state_method=STATE_METHOD, distance=DISTANCE, ablation_mode=param['ablation_mode'])
    if not os.path.exists('./tmp/'):
        os.mkdir('./tmp/')
    D_OPT_PATH = './tmp/' + NAME + '_' + \
                 MODE + '/'
    if NAME == 'fetal_health' or NAME == 'cardio_train' or NAME == 'breast_cancer' or NAME == 'alzheimers':
        data_path = os.path.join(base_path, NAME + '.csv')
    else:
        data_path = os.path.join(base_path, NAME + '.hdf')
    info('read the data from {}'.format(data_path))
    SAMPLINE_METHOD = param['replay_strategy']
    assert SAMPLINE_METHOD in REPLAY

    WE = 0.05 * param['intrinsic_weight']
    WD = 40 * param['steps']

    if NAME == 'fetal_health' or NAME == 'cardio_train' or NAME == 'breast_cancer' or NAME == 'alzheimers':
        Dg = pd.read_csv(data_path)
    else:
        Dg = pd.read_hdf(data_path)

    feature_names = list(Dg.columns)
    print(feature_names)
    info('initialize the features...')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = Dg.values[:, :-1]
    X = scaler.fit_transform(X)
    y = Dg.values[:, -1]
    Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    print(feature_names)
    Dg.columns = [str(i) for i in feature_names]
    D_OPT = Dg.copy()
    hidden = param['hidden_size']
    if SAMPLINE_METHOD == 'intrinsicPer':
        PRI_WEIGHT = param['priority_weight']
    else:
        PRI_WEIGHT = 0
    REPLACE_METHOD = param['replace_strategy']

    OP_DIM = len(operation_set)
    STATE_DIM = 0
    if ENV.state_method.__contains__('ae'):
        STATE_DIM += X.shape[0]
    if ENV.state_method.__contains__('cg'):
        STATE_DIM += X.shape[0]
    if ENV.state_method.__contains__('ds'):
        STATE_DIM += hidden
    mem_1_dim = STATE_DIM
    mem_2_dim = STATE_DIM + OP_DIM
    mem_op_dim = STATE_DIM
    info(f'initial memories with {SAMPLINE_METHOD}')
    BATCH_SIZE = param['batch_size']
    MEMORY_CAPACITY = param['memory']
    ENV.report_performance(Dg, D_OPT)
    if SAMPLINE_METHOD == 'random':
        cluster1_mem = RandomClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_1_dim, cuda_info)
        cluster2_mem = RandomClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_2_dim, cuda_info)
        op_mem = RandomOperationReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_op_dim, cuda_info)
    elif SAMPLINE_METHOD == 'per' or 'intrinsicPer':
        cluster1_mem = PERClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_1_dim, cuda_info, REPLACE_METHOD)
        cluster2_mem = PERClusterReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_2_dim, cuda_info, REPLACE_METHOD)
        op_mem = PEROperationReplay(MEMORY_CAPACITY, BATCH_SIZE, mem_op_dim, cuda_info, REPLACE_METHOD)
    else:
        error(f'unsupported sampling method {SAMPLINE_METHOD}')
        assert False
    ENT_WEIGHT = param['ent_weight']
    LR = 0.01
    init_w = param['init_w']
    hyperparameter_defaults = param
    hyperparameter_defaults.update({'lr':LR})

    model_cluster1 = ClusterActorCritic(state_dim=STATE_DIM, cluster_state_dim=STATE_DIM, hidden_dim=STATE_DIM * 2,
                                        memory=cluster1_mem,
                                        ent_weight=ENT_WEIGHT, select='head',
                                        gamma=0.99, state_method=STATE_METHOD,
                                        device=cuda_info, init_w=init_w)

    model_cluster2 = ClusterActorCritic(state_dim=STATE_DIM, cluster_state_dim=STATE_DIM, hidden_dim=(STATE_DIM + OP_DIM) * 2,
                                        memory=cluster2_mem,
                                        ent_weight=ENT_WEIGHT, select='tail',
                                        gamma=0.99, state_method=STATE_METHOD,
                                        device=cuda_info, init_w=init_w)

    model_op = OpActorCritic(state_dim=STATE_DIM, cluster_state_dim=STATE_DIM, hidden_dim=STATE_DIM * 2,
                             memory=op_mem, ent_weight=ENT_WEIGHT, state_method=STATE_METHOD, gamma=0.99, device=cuda_info, init_w=init_w)
    if cuda_info:
        model_cluster1 = model_cluster1.cuda()
        model_cluster2 = model_cluster2.cuda()
        model_op = model_op.cuda()
    optimizer_op = torch.optim.Adam(model_op.parameters(), lr=LR)
    optimizer_c2 = torch.optim.Adam(model_cluster2.parameters(), lr=LR)
    optimizer_c1 = torch.optim.Adam(model_cluster1.parameters(), lr=LR)


    EPISODES = param['episodes']
    STEPS = param['steps']
    episode = 0

    old_per, _, _, _ = ENV.get_reward(Dg, episode, 0, feature_path, performance_path, list(Dg.columns), NAME, MODE)
    best_per = old_per
    base_per = old_per
    info(f'start training, the original performance is {old_per}')
    D_original = Dg.copy()
    steps_done = 0
    FEATURE_LIMIT = Dg.shape[1] * param['enlarge_num']
    best_step = -1
    best_episode = -1
    training_start_time = time.time()
    max_step = STEPS * 10 # for each episode, we made 10 times try because some numerical error
    original_max = min(max(D_original.abs().max())* 1e10, 1e15) # numerical limitation to aviod some error
    features = []
    performances = []
    feature_lists = []
    performance_lists = []
    while episode < EPISODES:
        step = 0
        real_step = 0
        Dg = D_original.copy()
        best_per_opt = []
        fatal_time = 0 # record the fatal time
        while step < max_step: # at least try max_step
            intrinsic_reward = 0
            Dg_previous = Dg.copy(True)
            if real_step >= STEPS:
                break
            step_start_time = time.time()
            clusters = ENV.cluster_build(Dg.values[:, :-1], Dg.values[:, -1], cluster_num=3)

            acts1, action_emb, f_names1, f_cluster1, action_list, state_emb = \
                model_cluster1.select_action(clusters=clusters, X=Dg.values[:, :-1], feature_names=feature_names, steps_done=steps_done)
            op, op_name = model_op.select_operation(action_emb, steps_done=steps_done)
            if op_name in O1:
                Dg, is_op = model_cluster1.op(Dg, f_cluster1, f_names1, op_name)
            else:
                acts2, action_emb2, f_names2, f_cluster2, _, state_emb2 = \
                    model_cluster2.select_action(clusters, Dg.values[:, :-1], feature_names,
                                                 op_name, cached_state_embed=state_emb, cached_cluster_state=action_list, steps_done=steps_done)
                if FEATURE_LIMIT * 4 < (f_cluster1.shape[1] * f_cluster2.shape[1]):
                    is_op = False
                else:
                    Dg, is_op = model_cluster1.op(Dg, f_cluster1, f_names1, op_name, f_cluster2, f_names2)
            
            valid_move = False
            
            if not is_op: # current operation make wrong move
                Dg = Dg_previous
                fatal_time += 1
                info(f'model make a wrong move, now the fatal time is {fatal_time}')
                reward = param['operation_error_penalty']
                new_per = 0.0
                r_c1, r_op, r_c2 = param['a'] * reward, param['b'] * reward, param['c'] * reward
                if fatal_time > 15:
                    break
            elif max(Dg.abs().max()) > original_max:
                Dg = Dg_previous
                # in numrical error state!
                new_per = 0.0
                info(f'model make a numerical error, retreat to previous')
                reward = param['numerical_error_penalty']
                r_c1, r_op, r_c2 = param['a'] * reward, param['b'] * reward, param['c'] * reward
 
            else: # normal ! done one step
                if step == 0 and episode % 5 == 0:
                    feature_lists_1, performances_1 = model_cluster1.memory.retrain_sample()
                    feature_lists_2, performances_2 = model_cluster2.memory.retrain_sample()
                    feature_lists_0, performances_0 = model_op.memory.retrain_sample()
                    feature_lists = feature_lists_1 + feature_lists_2 + feature_lists_0
                    performance_lists = performances_1 + performances_2 + performances_0
                fatal_time = 0
                valid_move = True
                feature_names = list(Dg.columns)
                intrinsic_reward = 0
                new_per, intrinsic_reward, train_time, estime = ENV.get_reward(Dg, episode, step, feature_path, performance_path, list(Dg.columns), NAME, MODE, feature_lists, performance_lists)
                print('intrinsic reward:', intrinsic_reward)
                print(new_per)
                print(old_per)

                if MODE == 'lstmrnd' or MODE == 'rnd':
                    weight = WE + (param['intrinsic_weight'] - WE) * math.exp(-1.0 * steps_done / WD)
                    weight = param['intrinsic_weight']
                    reward = new_per - old_per + weight * intrinsic_reward
                else:
                    reward = new_per - old_per
                r_c1, r_op, r_c2 = param['a'] * reward, param['b'] * reward, param['c'] * reward

                if new_per > best_per:
                    best_step = step
                    best_episode = episode
                    best_per = new_per
                    D_OPT = Dg.copy()
                    if not os.path.exists(D_OPT_PATH):
                        os.mkdir(D_OPT_PATH)
                    out_name = 'best.csv'
                    D_OPT.to_csv(os.path.join(D_OPT_PATH, out_name))

                old_per = new_per
            
            
            
            select_time = time.time()
            total_cost_step = select_time - step_start_time
            info(f'this steps cost time {total_cost_step}')
            clusters_ = ENV.cluster_build(Dg.values[:, :-1], Dg.values[:, -1], cluster_num=3)
            acts_, action_emb_, f_names1_, f_cluster1_, action_list_, state_emb_ = \
                model_cluster1.select_action(clusters_, Dg.values[:, :-1], feature_names, for_next=True)
            op_, op_name_ = model_op.select_operation(action_emb_, for_next=True)
            if op_name in O2:
                _, _, _, _, _, state_emb2_ = \
                    model_cluster2.select_action(clusters_, Dg.values[:, :-1], feature_names,
                                                 op=op_name_, cached_state_embed=state_emb_,
                                                 cached_cluster_state=action_list_, for_next=True)
                model_cluster2.store_transition(state_emb2, acts2, r_c2, state_emb2_, action_list, intrinsic_reward, PRI_WEIGHT, feature_names, new_per)
            model_cluster1.store_transition(state_emb, acts1, r_c1, state_emb_, action_list, intrinsic_reward, PRI_WEIGHT, feature_names, new_per)
            model_op.store_transition(action_emb, op, r_op, action_emb_, intrinsic_reward, PRI_WEIGHT, feature_names, new_per)
            
            train_time = time.time()
            train_ = False
            if model_cluster1.memory.memory_counter >= model_cluster1.memory.MEMORY_CAPACITY:
                train_ = True
                info('start to learn in model_c1')
                model_cluster1.learn(optimizer_c1)
            if model_cluster2.memory.memory_counter >= model_cluster2.memory.MEMORY_CAPACITY:
                train_ = True
                info('start to learn in model_c2')
                model_cluster2.learn(optimizer_c2)
            if model_op.memory.memory_counter >= model_op.memory.MEMORY_CAPACITY:
                train_ = True
                info('start to learn in model_op')
                model_op.learn(optimizer_op)

            if Dg.shape[1] > FEATURE_LIMIT:
                fs_time = time.time()
                selector = SelectKBest(mutual_info_regression, k=int(FEATURE_LIMIT / param['shrink']))\
                    .fit(Dg.iloc[:, :-1], Dg.iloc[:, -1])
                cols = selector.get_support()
                X_new = Dg.iloc[:, :-1].loc[:, cols]
                Dg = pd.concat([X_new, Dg.iloc[:, -1]], axis=1)
            info('New performance is: {:.6f}, Best performance is: {:.6f} (e{}s{}) Base performance is: {:.6f}'
                .format(new_per, best_per, best_episode, best_step, base_per))
            info('Episode {}, Step {} ends!'.format(episode, step))
            best_per_opt.append(best_per)
            info('Current spend time for step-{} is: {:.1f}s'.format(step,
                                                                     time.time() - step_start_time))
            step += 1
            if valid_move:
                feature_list = list(Dg.columns)
                features.append(feature_list)
                performances.append(new_per)
                steps_done += 1
                real_step += 1
        if episode % 5 == 0:
            print(best_per_opt)
            info('Best performance is: {:.6f}'.format(np.max(best_per_opt)))
            info('Episode {} ends!'.format(episode))
            with open(feature_path + 'feature'+ str(episode)+'.pkl', 'wb') as file:
                pickle.dump(features, file)
            with open(performance_path + 'performance' + str(episode)+'.pkl', 'wb') as file:
                pickle.dump(performances, file)
        episode += 1
    with open(feature_path + 'feature'+ str(episode)+'.pkl', 'wb') as file:
        pickle.dump(features, file)
    with open(performance_path + 'performance' + str(episode)+'.pkl', 'wb') as file:
        pickle.dump(performances, file)
    info('Total training time for is: {:.1f}s'.format(time.time() -
                                                      training_start_time))
    info('Exploration ends!')
    info('Begin evaluation...')
    final = ENV.report_performance(D_original, D_OPT)
    info('Total using time: {:.1f}s'.format(time.time() - training_start_time))


if __name__ == '__main__':

    try:
        args = init_param()
        args = vars(args)
        info(args)
        train(args)
    except Exception as exception:
        error(exception)
        raise
