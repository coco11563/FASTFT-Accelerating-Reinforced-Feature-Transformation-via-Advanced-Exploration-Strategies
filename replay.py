from collections import defaultdict

import numpy as np
import torch


class Replay:
    def __init__(self, size, batch_size, device):
        self.MEMORY_CAPACITY = size
        self.memory_counter = 0
        self.BATCH_SIZE = batch_size
        self.cuda_info = device is not None

    def _sample(self):
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        return sample_index

    def sample(self):
        raise NotImplementedError()

    def store_transition(self, resource):
        raise NotImplementedError()


class RandomClusterReplay(Replay):
    def __init__(self, size, batch_size, state_shape, device):
        super().__init__(size, batch_size, device)
        self.mem1 = torch.zeros(self.MEMORY_CAPACITY, state_shape)
        self.mem2 = torch.zeros(self.MEMORY_CAPACITY, state_shape)
        self.action = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.reward = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.featurelists = [None] * self.MEMORY_CAPACITY
        self.performances = [None] * self.MEMORY_CAPACITY
        if self.cuda_info:
            self.mem1 = self.mem1.cuda()
            self.mem2 = self.mem2.cuda()
            self.action = self.action.cuda()
            self.reward = self.reward.cuda()
        self.alist = [None] * self.MEMORY_CAPACITY

    def store_transition(self, mems):
        s1, action, r, s2, alist, feature_list, performance = mems
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.mem1[index] = s1
        self.action[index] = action
        self.mem2[index] = s2
        self.reward[index] = r
        self.alist[index] = alist
        self.featurelists[index] = feature_list
        self.performances[index] = performance
        self.memory_counter += 1
    
    def retrain_sample(self):
        print('feature_lists_count:', len(self.featurelists))
        print('performance_lists_count:', len(self.performances))
        return self.featurelists, self.performances

    def sample(self):
        sample_index = self._sample()
        alist = [self.alist[i] for i in sample_index]
        return self.mem1[sample_index], self.action[sample_index], self.reward[sample_index], self.mem2[sample_index], alist


class RandomOperationReplay(Replay):
    def __init__(self, size, batch_size, state_dim, device):
        super().__init__(size, batch_size, device)
        self.STATE_DIM = state_dim
        self.mem1 = torch.zeros(self.MEMORY_CAPACITY, self.STATE_DIM)
        self.mem2 = torch.zeros(self.MEMORY_CAPACITY, self.STATE_DIM)
        self.op = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.reward = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.featurelists = [None] * self.MEMORY_CAPACITY
        self.performances = [None] * self.MEMORY_CAPACITY
        if self.cuda_info:
            self.mem1 = self.mem1.cuda()
            self.mem2 = self.mem2.cuda()
            self.op = self.op.cuda()
            self.reward = self.reward.cuda()

    def store_transition(self, mems):
        s1, op, r, s2, feature_list, performance = mems
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.mem1[index] = s1
        self.mem2[index] = s2
        self.reward[index] = r
        self.op[index] = op
        self.featurelists[index] = feature_list
        self.performances[index] = performance
        self.memory_counter += 1

    def retrain_sample(self):
        print('feature_lists_count:', len(self.featurelists))
        print('performance_lists_count:', len(self.performances))
        return self.featurelists, self.performances
    
    def sample(self):
        sample_index = self._sample()
        return self.mem1[sample_index], self.op[sample_index], self.reward[sample_index], self.mem2[sample_index]


class SumTree:
    def __init__(self, capacity, replace):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * self.capacity
        self.chances = np.zeros(capacity)
        self.maxchances = 3
        self.data_pointer = 0
        self.n_entries = 0
        self.replace = replace

    def add(self, priority, data):
        if self.replace == 'direct':
            tree_index = self.data_pointer + self.capacity - 1
            self.data[self.data_pointer] = data
            self.update(tree_index, priority)

            self.data_pointer += 1
            if self.data_pointer >= self.capacity:
                self.data_pointer = 0

            if self.n_entries < self.capacity:
                self.n_entries += 1
            return data

        elif self.replace == 'priority':
            if self.n_entries < self.capacity:
                tree_index = self.data_pointer + self.capacity - 1
                self.data[self.data_pointer] = data
                self.update(tree_index, priority)
                self.data_pointer += 1
                self.n_entries += 1
                return data
            else:
                argmin = self.compare(priority)
                if argmin < 0:
                    return -1
                else:
                    index = self.data[argmin]
                    self.data[argmin] = data
                    self.update(argmin + self.capacity - 1, priority)
                    return index
                    
        elif self.replace == 'compromise':
            if self.n_entries < self.capacity:
                tree_index = self.data_pointer + self.capacity - 1
                self.data[self.data_pointer] = data
                self.update(tree_index, priority)
                self.data_pointer += 1
                self.n_entries += 1
                return data
            else:
                argmin = self.chance_compare(priority)
                if argmin < 0:
                    return -1
                else:
                    index = self.data[argmin]
                    self.data[argmin] = data
                    self.chances[argmin] = 0
                    self.update(argmin + self.capacity - 1, priority)
                    return index

    def chance_compare(self, priority):
        flag = self.compare(priority)
        if flag >= 0:
            return flag
        arg = []
        ps = []
        for i, p in enumerate(self.tree[-self.capacity:]):
            if self.chances[i] < 3:
                continue
            else:
                arg.append(i)
                ps.append(p)
        if arg == []:
            return -1
        else:
            return arg[np.argmin(ps)]

    def compare(self, priority):
        minp, argmin = self.min_priority()
        if priority < minp:
            return -1
        else:
            return argmin

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        print("---------------------------------------")
        print(",,,", priority)
        self._propagate(tree_index, change)

    def _propagate(self, tree_index, change):
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1 
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        self.chances[data_index] += 1
        return self.tree[leaf_index], self.data[data_index]
    
    def total(self):
        return self.tree[0]

    def max_priority(self):
        return np.max(self.tree[-self.capacity:])
    
    def min_priority(self):
        return np.min(self.tree[-self.capacity:]), np.argmin(self.tree[-self.capacity:])

class PERClusterReplay(Replay):
    def __init__(self, size, batch_size, state_shape, device, replace, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=0.02, clip_error=False):
        super().__init__(size, batch_size, device)
        self.mem1 = torch.zeros(self.MEMORY_CAPACITY, state_shape)
        self.mem2 = torch.zeros(self.MEMORY_CAPACITY, state_shape)
        self.action = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.reward = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.alpha = alpha  # Prioritization parameter
        self.beta = beta  # Importance sampling parameter
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon  # Small positive constant to ensure all transitions have a non-zero probability of being sampled
        self.clip_error = clip_error
        self.replace = replace
        self.tree = SumTree(self.MEMORY_CAPACITY, self.replace)
        self.feature_lists = [None] * self.MEMORY_CAPACITY
        self.performances = [None] * self.MEMORY_CAPACITY
        if self.cuda_info:
            self.mem1 = self.mem1.cuda()
            self.mem2 = self.mem2.cuda()
            self.action = self.action.cuda()
            self.reward = self.reward.cuda()
        self.alist = [None] * self.MEMORY_CAPACITY
        

    def store_transition(self, mems, error=None):
        s1, action, r, s2, alist, feature_list, performance = mems
        index = self.memory_counter % self.MEMORY_CAPACITY

        if error is None:
            priority = self.tree.max_priority()
            if priority == 0:
                priority = 1  # Ensure all transitions have a non-zero probability of being sampled
        else:
            priority = self._get_priority(error)

        index = self.tree.add(priority, index)
        if index != -1:
            self.mem1[index] = s1
            self.action[index] = action
            self.mem2[index] = s2
            self.reward[index] = r
            self.alist[index] = alist
            self.feature_lists[index] = feature_list
            self.performances[index] = performance
            self.memory_counter += 1

    def _get_priority(self, error):
        if self.clip_error:
            error = min(abs(error), self.clip_error)
        error = abs(error)
        return (error + self.epsilon) ** self.alpha
    
    def sample(self):
        segment = self.tree.total() / self.BATCH_SIZE

        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        sample_index = []
        for i in range(self.BATCH_SIZE):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            priority, data_index = self.tree.get(value)
            sample_index.append(data_index)
            priorities.append(priority)
        sample_index = np.array(sample_index)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        alist = [self.alist[i] for i in sample_index]
        return is_weight, self.mem1[sample_index], self.action[sample_index], self.reward[sample_index], self.mem2[sample_index], alist
    
    def retrain_sample(self):
        print('feature_lists_count:', len(self.feature_lists))
        print('performance_lists_count:', len(self.performances))
        return self.feature_lists, self.performances

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # Add a small value to ensure non-zero priority
        clipped_errors = np.minimum(abs_errors, self.clip_error) if self.clip_error else abs_errors
        priorities = np.power(clipped_errors, self.alpha)

        for ti, priority in zip(tree_idx, priorities):
            self.tree.update(ti, priority)


class PEROperationReplay(Replay):
    def __init__(self, size, batch_size, state_dim, device, replace, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=0.01, clip_error=False):
        super().__init__(size, batch_size, device)
        self.STATE_DIM = state_dim
        self.mem1 = torch.zeros(self.MEMORY_CAPACITY, self.STATE_DIM)
        self.mem2 = torch.zeros(self.MEMORY_CAPACITY, self.STATE_DIM)
        self.op = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.reward = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.alpha = alpha  # Prioritization parameter
        self.beta = beta  # Importance sampling parameter
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon  # Small positive constant to ensure all transitions have a non-zero probability of being sampled
        self.clip_error = clip_error
        self.replace = replace
        self.tree = SumTree(self.MEMORY_CAPACITY, self.replace)
        if self.cuda_info:
            self.mem1 = self.mem1.cuda()
            self.mem2 = self.mem2.cuda()
            self.op = self.op.cuda()
            self.reward = self.reward.cuda()
        self.feature_lists = [None] * self.MEMORY_CAPACITY
        self.performances = [None] * self.MEMORY_CAPACITY

    def store_transition(self, mems, error=None):
        s1, op, r, s2, feature_list, performance = mems
        index = self.memory_counter % self.MEMORY_CAPACITY

        if error is None:
            priority = self.tree.max_priority()
            if priority == 0:
                priority = 1  # Ensure all transitions have a non-zero probability of being sampled
        else:
            priority = self._get_priority(error)
            
        index = self.tree.add(priority, index)
        if index != -1:
            self.mem1[index] = s1
            self.mem2[index] = s2
            self.reward[index] = r
            self.op[index] = op
            self.feature_lists[index] = feature_list
            self.performances[index] = performance
            self.memory_counter += 1


    def _get_priority(self, error):
        if self.clip_error:
            error = min(abs(error), self.clip_error)
        error = abs(error)
        return (error + self.epsilon) ** self.alpha

    def sample(self):
        segment = self.tree.total() / self.BATCH_SIZE

        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        sample_index = []
        for i in range(self.BATCH_SIZE):
            a = segment * i
            b = segment * (i + 1)

            value = np.random.uniform(a, b)
            priority, data_index = self.tree.get(value)
            sample_index.append(data_index)
            priorities.append(priority)
        sample_index = np.array(sample_index)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return is_weight, self.mem1[sample_index], self.op[sample_index], self.reward[sample_index], self.mem2[sample_index]
    
    def retrain_sample(self):
        print('feature_lists_count:', len(self.feature_lists))
        print('performance_lists_count:', len(self.performances))
        return self.feature_lists, self.performances

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # Add a small value to ensure non-zero priority
        clipped_errors = np.minimum(abs_errors, self.clip_error) if self.clip_error else abs_errors
        priorities = np.power(clipped_errors, self.alpha)

        for ti, priority in zip(tree_idx, priorities):
            self.tree.update(ti, priority)
