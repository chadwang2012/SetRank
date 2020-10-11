# -*- Encoding:UTF-8 -*-
# This is the implementation for Deep-SetRank using sparse matrix.

import tensorflow as tf
import numpy as np
import argparse
from DataSet_pre import DataSet
import os
import pandas as pd
from evaluate import *
from collections import defaultdict

tf.set_random_seed(1)
np.random.seed(1)

def main():
    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument('-dev', action='store', dest='dev', default='0')
    parser.add_argument('-trainName', action='store', dest='trainName', default='ml1m_oc_50_train_ratings')
    parser.add_argument('-testName', action='store', dest='testName', default='ml1m_oc_50_test_ratings')
    parser.add_argument('-repath', action='store', dest='repath', default='model/set.ckpt')
    parser.add_argument('-reloop', type=str, default=199)
    parser.add_argument('-negnum', action='store', dest='negnum', default=30, type=int)         # The number of sampled unobserved items in one epoch for each user
    parser.add_argument('-posnum', action='store', dest='posnum', default=20, type=int)         # The number of positive items in one epoch for each user
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 100])     # The shape of user network
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 100])    # The shape of item network
    parser.add_argument('-reg', action='store', dest='reg', default=1.2, type=np.float64)       # The regularization parameter
    parser.add_argument('-lr', action='store', dest='lr', default=0.0003, type=np.float64)      # The learning rate
    parser.add_argument('-keep', action='store', dest='keep', default=0.5, type=np.float64)     # The keep ratio in dropout layer
    parser.add_argument('-maxepochs', action='store', dest='maxEpochs', default=6000, type=int)
    parser.add_argument('-print', action='store', dest='print', default=1, type=int)

    args = parser.parse_args()
    classifier = Model(args)
    classifier.run()
    # restore_path = args.repath + '-%s' % args.reloop
    # classifier.saver.restore(classifier.sess, restore_path)
    # classifier.evaluate(classifier.sess)
    # classifier.run_epoch(classifier.sess)


def sparse_dropout(x, keep_prob, noise_shape):  # The dropout layer for sparse matrix
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([noise_shape], dtype=tf.float64)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


recall = []
precision = []
map = []


class Model:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.dev
        self.dataSet = DataSet(args)
        self.nrow = self.dataSet.nrow
        self.ncol = self.dataSet.ncol
        self.shape = [self.nrow,self.ncol]
        self.train_list = self.dataSet.train_list
        self.test_list = self.dataSet.test
        self.id_matrix = tf.convert_to_tensor(self.dataSet.id_matrix)
        self.posprobe = defaultdict(list)
        for line in self.test_list:
            self.posprobe[line[0]].append(line[1])
        self.postrain = defaultdict(list)
        for line in self.train_list:
            self.postrain[line[0]].append(line[1])
        self.negNum = args.negnum
        self.reg = args.reg
        self.keep = args.keep
        self.add_embedding_matrix()

        self.add_placeholders()
        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.add_model()

        self.add_loss()

        self.lr = args.lr
        self.add_train_step()

        self.filename = "dumper/deep-setrank" + "_" + str(self.keep) + "_" + str(args.negnum) + "_" + str(self.reg) + "_" + str(self.lr) + "_" + str(self.itemLayer[-1])
        self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.print = args.print
        self.repath = args.repath

    def add_placeholders(self):
        self.user = tf.placeholder(tf.int64)
        self.item = tf.placeholder(tf.int64)
        self.item2 = tf.placeholder(tf.int64)
        self.drop = tf.placeholder(tf.float64)

    def add_embedding_matrix(self):
        self.user_item_embedding = tf.SparseTensor(self.dataSet.train_list[:, 0:2], values=self.dataSet.train_list[:, 2].astype(np.float64), dense_shape=[self.nrow, self.ncol])
        self.item_user_embedding = tf.sparse_transpose(self.user_item_embedding)

    def add_model(self):
        # user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        user_input = self.user_item_embedding
        item_input = self.item_user_embedding
        self.id_input = tf.nn.embedding_lookup(self.id_matrix, self.user)
        self.n_id = tf.reduce_sum(self.id_input)
        user_input = sparse_dropout(user_input, self.drop, self.dataSet.n_train)
        item_input = sparse_dropout(item_input, self.drop, self.dataSet.n_train)

        def init_variable(shape, name):
            return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            user_b1 = tf.get_variable("user_b1", [self.userLayer[0]], initializer=tf.constant_initializer(0.0), dtype=tf.float64)
            self.user_out = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(user_input, user_W1) + user_b1)
            for i in range(0, len(self.userLayer)-1):
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                b = tf.get_variable("user_b"+str(i+2), [self.userLayer[i+1]], initializer=tf.constant_initializer(0.0), dtype=tf.float64)
                self.user_out = tf.nn.tanh(tf.add(tf.matmul(self.user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_b1 = tf.get_variable("item_b1" , [self.itemLayer[0]], initializer=tf.constant_initializer(0.0), dtype=tf.float64)
            self.item_preout = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(item_input, item_W1) + item_b1)
            for i in range(0, len(self.itemLayer)-1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                b = tf.get_variable("item_b"+str(i+2), [self.itemLayer[i+1]], initializer=tf.constant_initializer(0.0), dtype=tf.float64)
                self.item_preout = tf.nn.tanh(tf.add(tf.matmul(self.item_preout, W), b))

        self.user_tile = tf.reshape(tf.tile(self.user_out, [1, self.dataSet.maxn_item]), (-1, self.itemLayer[-1]))
        self.user_tile2 = tf.reshape(tf.tile(self.user_out, [1, self.negNum]), (-1, self.itemLayer[-1]))
        self.item_out = tf.nn.embedding_lookup(self.item_preout, self.item)
        self.item2_out = tf.nn.embedding_lookup(self.item_preout, self.item2)

        self.y = tf.reduce_sum(tf.multiply(self.user_tile, self.item_out), axis=1)
        self.y2 = tf.reduce_sum(tf.multiply(self.user_tile2, self.item2_out), axis=1)
        self.y_ = tf.reshape(tf.nn.sigmoid(self.y), (-1, self.dataSet.maxn_item))
        self.y2_ = tf.reshape(tf.nn.sigmoid(self.y2), (-1, self.negNum))

        self.y_sum = tf.reshape(tf.reduce_sum(self.y2_, axis=1), (-1, 1)) + self.y_

    def add_loss(self):
        self.model_loss = - tf.reduce_mean(tf.reduce_sum(tf.log(tf.maximum(self.y_ / self.y_sum, 1e-9)) * self.id_input, 1))
        self.norm_loss = self.reg * (tf.nn.l2_loss(self.user_out)/self.nrow + tf.nn.l2_loss(self.item_preout)/self.ncol)
        self.loss = self.model_loss + self.norm_loss

    def add_train_step(self):
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.train.Saver(self.weights, max_to_keep=30)

    def run(self):
        print("Start Training!")
        for epoch in range(self.maxEpochs):
            print("=" * 20 + "Epoch ", epoch, "=" * 20)
            if epoch % self.print == 0:
                self.evaluate(self.sess)
            # if epoch % 100 == 0:
            #     self.saver.save(self.sess, self.repath, global_step=epoch)
            self.run_epoch(self.sess)

    def run_epoch(self, sess):
        train_u, train_i, train_j = self.dataSet.getInstances(self.negNum)
        losses = []
        losses2 = []
        losses1 = []

        feed_dict = self.create_feed_dict(train_u, np.reshape(train_i, (-1)), np.reshape(train_j, (-1)), self.keep)
        _, tmp_loss, loss1, loss2 = sess.run([self.train_step, self.loss, self.model_loss, self.norm_loss], feed_dict=feed_dict)
        losses.append(tmp_loss)
        losses1.append(loss1)
        losses2.append(loss2)

        loss = np.mean(losses)
        loss1 = np.mean(losses1)
        loss2 = np.mean(losses2)
        print("\nMean loss in this epoch is: {}".format(loss), loss1, loss2)
        return loss

    def create_feed_dict(self, u, i, j, drop=None):
        return {self.user: u,
                self.item: i,
                self.item2: j,
                self.drop: drop}

    def evaluate(self, sess):
        testUser1 = np.arange(self.nrow)
        testItem1 = np.arange(self.ncol)
        self.u, self.v = self.sess.run((self.user_out, self.item_out), feed_dict={self.user: testUser1, self.item: testItem1, self.drop: 1})
        epoch_rating = np.dot(self.u, self.v.T)
        recall_batch, precision_batch, map_batch = evaluate(self.postrain, self.posprobe, epoch_rating, [1, 5, 10, 20])
        print(precision_batch, recall_batch, map_batch)
        precision.append(precision_batch)
        recall.append(recall_batch)
        map.append(map_batch)
        evaluation = pd.concat([pd.DataFrame(precision), pd.DataFrame(recall), pd.DataFrame(map)], axis=1)
        evaluation.to_csv(self.filename + ".csv", header=False, index=False)


if __name__ == '__main__':
    main()
