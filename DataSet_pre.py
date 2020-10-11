# -*- Encoding:UTF-8 -*-
# This is the data preprocessing for Deep-SetRank.

import numpy as np
import pandas as pd
from collections import defaultdict


class DataSet(object):
    def __init__(self, args):
        self.train, self.test, self.train_list = self.load_data(args.trainName, args.testName)
        self.nrow, self.ncol = np.max(self.train_list[:, 0:2], axis=0) + 1
        self.shape = [self.nrow, self.ncol]
        self.n_train = len(self.train_list)
        self.user_ratings = defaultdict(list)
        self.nuser_item = np.zeros(self.nrow, dtype=np.int64)
        for line in self.train_list:
            self.user_ratings[line[0]].append(line[1])
            self.nuser_item[line[0]] += 1
        self.maxn_item = np.minimum(np.max(self.nuser_item), args.posnum)
        self.id_matrix = np.ones((self.nrow, self.maxn_item), dtype=np.float64)
        for i in range(self.nrow):
            self.id_matrix[i, self.nuser_item[i]:] = 0
        self.train_matrix, self.trainDict = self.getEmbedding()
        self.item_all = np.array(np.arange(self.ncol))
        self.sample_prob = 1 - self.train_matrix
        self.sample_prob = self.sample_prob / np.sum(self.sample_prob, axis=1, keepdims=True)

    def load_data(self, trainName, testName):
        data = {}
        data["test_list"] = pd.read_csv("data/" + testName + ".csv", header=None).values
        data["train_list"] = pd.read_csv("data/" + trainName + ".csv", header=None).values
        data["test_list"][:, 0:2] = data["test_list"][:, 0:2] - 1
        data["train_list"][:, 0:2] = data["train_list"][:, 0:2] - 1
        train = []
        for i in data["train_list"]:
            if i[2] == 1:
                train.append((i[0], i[1], 1.0))
        return train, data["test_list"], data["train_list"]

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float64)
        dataDict = {}
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
            dataDict[(i[0], i[1])] = i[2]
        return np.array(train_matrix), dataDict

    def getInstances(self, negNum):  # 选部分正样本，随机负样本
        item_pos = np.zeros((self.nrow, self.maxn_item), dtype=np.int64)
        item_neg = np.zeros((self.nrow, negNum), dtype=np.int64)
        for i in range(self.nrow):
            if self.user_ratings[i]:
                temp_pos = np.random.choice(self.user_ratings[i], size=np.minimum(self.maxn_item, len(self.user_ratings[i])), replace=False)
                item_pos[i, :] = np.pad(temp_pos, (0, self.maxn_item - len(temp_pos)), 'constant')
            item_neg[i, :] = np.random.choice(self.item_all, size=negNum, replace=False, p=self.sample_prob[i, :])
        return np.array(range(self.nrow)), item_pos, item_neg


