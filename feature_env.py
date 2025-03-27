"""
feature env
interactive with the actor critic for the state and state after action
"""
from collections import namedtuple

from utils.logger import error, info
from utils.tools import feature_state_generation, downstream_task_new, test_task_new, cluster_features

import LSTM
import lstm_rnd
import time
import numpy as np
import statistics

TASK_DICT = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg',
             'smtp': 'det', 'thyroid': 'det', 'yeast': 'det', 'wbc': 'det', 'mammography': 'det',
             'fetal_health':'cls', 'cardio_train':'cls', 'breast_cancer':'cls', 'diabetes':'cls',
             'cardio_10':'cls','cardio_20':'cls','cardio_30':'cls','cardio_40':'cls','cardio_50':'cls',
             'cardio_60':'cls','cardio_70':'cls','cardio_80':'cls','cardio_90':'cls','cardio_100':'cls', 
             'breast_cancer_10':'cls', 'breast_cancer_20':'cls', 'blood':'cls', 'alzheimers':'cls'
             }

SUPPORT_STATE_METHOD = {
    'ds', 'ae', 'cg', 'ds+cg', 'ds+ae', 'ae+cg', 'ds+ae+cg'
}

MEASUREMENT = {
    'cls': ['precision', 'recall', 'f1_score'],
    'reg': ['mae', 'mse', 'rae'],
    'det': ['map', 'f1_score', 'ras']
}

REPLAY = {
    'random', 'per', 'intrinsicPer'
}


class FeatureEnv:
    def __init__(self, state_method, task_name, task_type=None, distance='mi', ablation_mode=''):
        if task_type is None:
            self.task_type = TASK_DICT[task_name]
        else:
            self.task_type = task_type
        if not state_method in SUPPORT_STATE_METHOD:
            error(f'Unsupported State Method {state_method} \n\tthe choice should in {SUPPORT_STATE_METHOD}')
            raise Exception('Unsupported state method for feature env')
        info(f'initial the feature env with state method {state_method}')
        self.state_method = state_method
        self.model_performance = namedtuple('ModelPerformance', MEASUREMENT[self.task_type])
        self.distance = distance
        self.predict_model = None
        self.discriminator = None
        self.train_flag = 0
        self.max_val = 0
        self.min_val = 0
        self.threshold_1 = 0
        self.threshold_2 = 0
        self.steps1 = []
        self.steps2 = []
        self.q = []
        if ablation_mode == '-c':
            self.mode = 'c'
        else:
            self.mode = ''

    '''
        input a Dataframe (cluster or feature set)
        :return the feature status
        return type is Numpy array
    '''

    def get_feature_state(self, data):
        return feature_state_generation(data, self.state_method)

    '''
        input a Dataframe (cluster or feature set)
        :return the current dataframe performance
        return type is Numpy array
    '''

    def predict_reward(self, feature_list, name):
        norm_val = LSTM.predict(self.predict_model, feature_list, name)
        print('norm_val:', norm_val)
        return norm_val


    def train_lstm(self, feature_path, performance_path, name):
        feature_path = feature_path + 'feature10.pkl'
        performance_path = performance_path + 'performance10.pkl'
        model, max_val, min_val, threshold = LSTM.train(feature_path, performance_path, name)
        if threshold > 1.0:
            threshold = 1.0
        return model, max_val, min_val, threshold
    
    def train_rnd(self, feature_path, performance_path, name):
        feature_path = feature_path + 'feature10.pkl'
        performance_path = performance_path + 'performance10.pkl'
        model, threshold = lstm_rnd.train(feature_path, performance_path, name)
        return model, threshold


    def get_reward(self, data, episode, step, feature_path, performance_path, feature_list, name, mode, feature_lists=None, performances=None):
        LSTM_train_time = 0
        if mode == 'lstmrnd':
            train_episode = 11
            if episode == train_episode:
                if self.train_flag == 0:
                    start_time = time.time()
                    self.predict_model, self.max_val, self.min_val, self.threshold_2 = self.train_lstm(feature_path, performance_path, name)
                    self.discriminator, self.threshold_1 = self.train_rnd(feature_path, performance_path, name)
                    lstm_time = time.time() - start_time
                    LSTM_train_time = lstm_time
                    self.train_flag = 1
                    return downstream_task_new(data, self.task_type), 0, LSTM_train_time, 0
                else:
                    return downstream_task_new(data, self.task_type), 0, LSTM_train_time, 0
            elif episode < train_episode:
                return downstream_task_new(data, self.task_type), 0, LSTM_train_time, 0
            else:
                start = time.time()
                if step == 0 and episode % 10 == 0:
                    s = time.time()
                    self.predict_model = LSTM.retrain(self.predict_model, name, feature_lists, performances)
                    self.discriminator = lstm_rnd.retrain(self.discriminator, name, feature_lists)
                    e = time.time()
                    LSTM_train_time = e - s
                self.discriminator.eval()
                self.predict_model.eval()
                feature = feature_list[-2]
                print("new_feature:", feature)
                

                flag_all = lstm_rnd.predict_all(self.discriminator, feature_list, name)
                intrinsic = flag_all * flag_all
                print("intrinsic reward:", intrinsic)
                self.steps1.append(abs(flag_all))
                if len(self.steps1) > 20:
                    self.threshold_1 = statistics.quantiles(self.steps1, n=20)[10]
                    if self.threshold_1 > 1:
                        self.threshold_1 = 1.0
                print('threshold_1:', self.threshold_1)
                if abs(flag_all) > self.threshold_1:
                    print("new feature is very new! Go to downstream task.")
                    end = time.time() - start
                    return downstream_task_new(data, self.task_type), intrinsic, LSTM_train_time, end
                else:
                    print("new feature is similar to the features before!")
                    p = self.predict_reward(feature_list, name).item()
                    self.steps2.append(p)
                    if len(self.steps2) > 20:
                        self.threshold_2 = statistics.quantiles(self.steps2, n=20)[10]
                        if self.threshold_2 > 1:
                            self.threshold_2 = 1.0
                    print('threshold_2:', self.threshold_2)
                    if p >= self.threshold_2:
                        print("predict model says present feature set may have high performance! Go to downstream task.")
                        end = time.time() - start
                        return downstream_task_new(data, self.task_type), intrinsic, LSTM_train_time, end
                    else:
                        print("predict model says present feature set have low performance! Use predict value.")
                        p = p * (self.max_val - self.min_val) + self.min_val
                        end = time.time() - start
                        return p, intrinsic, LSTM_train_time, end

        elif mode == 'lstm':
            train_episode = 11
            if episode == train_episode:
                if self.train_flag == 0:
                    start_time = time.time()
                    self.predict_model, self.max_val, self.min_val, self.threshold_2 = self.train_lstm(feature_path, performance_path, name)
                    self.discriminator, self.threshold_1 = self.train_rnd(feature_path, performance_path, name)
                    lstm_time = time.time() - start_time
                    LSTM_train_time = lstm_time
                    self.train_flag = 1
                    return downstream_task_new(data, self.task_type), 0, LSTM_train_time, 0
                else:
                    return downstream_task_new(data, self.task_type), 0, LSTM_train_time, 0
            elif episode < train_episode:
                return downstream_task_new(data, self.task_type), 0, LSTM_train_time, 0
            else:
                if step == 0 and episode % 10 == 0:
                    s = time.time()
                    self.predict_model = LSTM.retrain(self.predict_model, name, feature_lists, performances)
                    self.discriminator = lstm_rnd.retrain(self.discriminator, name, feature_lists)
                    LSTM_train_time = time.time() - s
                self.discriminator.eval()
                self.predict_model.eval()
                feature = feature_list[-2]
                print("new_feature:", feature)
                flag_all = lstm_rnd.predict_all(self.discriminator, feature_list, name)
                intrinsic = flag_all * flag_all
                p = self.predict_reward(feature_list, name).item()
                print('threshold_2:', self.threshold_2)
                if p >= self.threshold_2:
                    print("predict model says present feature set may have high performance! Go to downstream task.")
                    return downstream_task_new(data, self.task_type), intrinsic, LSTM_train_time, 0
                else:
                    print("predict model says present feature set have low performance! Use predict value.")
                    p = p * (self.max_val - self.min_val) + self.min_val
                    return p, intrinsic, LSTM_train_time, 0
                
        else:
            train_episode = 11
            if episode == train_episode:
                if self.train_flag == 0:
                    start_time = time.time()
                    self.discriminator, self.threshold_1 = self.train_rnd(feature_path, performance_path, name)
                    lstm_time = time.time() - start_time
                    LSTM_train_time = lstm_time
                    self.train_flag = 1
                    return downstream_task_new(data, self.task_type), 0, LSTM_train_time
                else:
                    return downstream_task_new(data, self.task_type), 0, LSTM_train_time
            elif episode <train_episode:
                return downstream_task_new(data, self.task_type), 0, LSTM_train_time
            else:
                if step == 0 and episode % 10 == 0:
                    s = time.time()
                    self.discriminator = lstm_rnd.retrain(self.discriminator, name, feature_lists)
                    LSTM_train_time = time.time() - s
                self.discriminator.eval()
                feature = feature_list[-2]
                print("new_feature:", feature)
                flag_all = lstm_rnd.predict_all(self.discriminator, feature_list, name)
                intrinsic = flag_all * flag_all
                print("intrinsic reward:", intrinsic)
                return downstream_task_new(data, self.task_type), intrinsic, LSTM_train_time
        

    '''
        input a Dataframe (cluster or feature set)
        :return the current dataframe performance on few dataset
        its related measure is listed in {MEASUREMENT[self.task_type]}
        return type is Numpy array
    '''

    def get_performance(self, data):
        a, b, c = test_task_new(data, task=self.task_type)
        return self.model_performance(a, b, c)

    def cluster_build(self, X, y, cluster_num):
        return cluster_features(X, y, cluster_num, mode=self.mode, distance_method=self.distance)

    def report_performance(self, original, opt):
        report = self.get_performance(opt)
        original_report = self.get_performance(original)
        if self.task_type == 'reg':
            final_result = report.rae
            info('MAE on original is: {:.3f}, MAE on generated is: {:.3f}'.
                 format(original_report.mae, report.mae))
            info('RMSE on original is: {:.3f}, RMSE on generated is: {:.3f}'.
                 format(original_report.mse, report.mse))
            info('1-RAE on original is: {:.3f}, 1-RAE on generated is: {:.3f}'.
                 format(original_report.rae, report.rae))
        elif self.task_type == 'cls':
            final_result = report.f1_score
            info('Pre on original is: {:.3f}, Pre on generated is: {:.3f}'.
                 format(original_report.precision, report.precision))
            info('Rec on original is: {:.3f}, Rec on generated is: {:.3f}'.
                 format(original_report.recall, report.recall))
            info('F-1 on original is: {:.3f}, F-1 on generated is: {:.3f}'.
                 format(original_report.f1_score, report.f1_score))
        elif self.task_type == 'det':
            final_result = report.ras
            info(
                'Average Precision Score on original is: {:.3f}, Average Precision Score on generated is: {:.3f}'
                .format(original_report.map, report.map))
            info(
                'F1 Score on original is: {:.3f}, F1 Score on generated is: {:.3f}'
                .format(original_report.f1_score, report.f1_score))
            info(
                'ROC AUC Score on original is: {:.3f}, ROC AUC Score on generated is: {:.3f}'
                .format(original_report.ras, report.ras))
        else:
            error('wrong task name!!!!!')
            assert False
        return final_result
