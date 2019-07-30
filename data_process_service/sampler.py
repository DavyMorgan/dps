#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import scipy.sparse as sp


class Sampler(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_sampler'
    
    def sample(self, index, **kwargs):

        raise NotImplementedError


class PointSampler(Sampler):

    def __init__(self, flags_obj, lil_record, dok_record, neg_sample_rate):

        super(PointSampler, self).__init__(flags_obj)
        self.lil_record = lil_record
        self.record = list(dok_record.keys())
        self.neg_sample_rate = neg_sample_rate
        self.n_user = lil_record.shape[0]
        self.n_item = lil_record.shape[1]
    
    def sample(self, index):

        user = self.record[index][0]
        pos_item = self.record[index][1]

        users = np.full(1 + self.neg_sample_rate, user, dtype=np.int64)
        items = np.full(1 + self.neg_sample_rate, pos_item, dtype=np.int64)
        labels = np.zeros(1 + self.neg_sample_rate, dtype=np.int64)
        
        labels[0] = 1

        user_pos = self.lil_record.rows[user]
        for count in range(self.neg_sample_rate):

            item = np.random.randint(self.n_item)
            while item in user_pos or item in items:
                item = np.random.randint(self.n_item)
            items[1+count] = item
        
        return users, items, labels
