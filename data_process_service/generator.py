#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Generator(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_generator'
    
    def generate(self, record):

        raise NotImplementedError
    
    def stat(self, record):

        self.n_record = len(record)
        self.n_user = record['uid'].nunique()
        self.n_item = record['iid'].nunique()


class LilGenerator(Generator):

    def __init__(self, flags_obj):

        super(LilGenerator, self).__init__(flags_obj)
    
    def generate(self, record):

        self.stat(record)

        values = np.ones(self.n_record)
        users = record['uid'].to_numpy()
        items = record['iid'].to_numpy()
        lil_record = sp.coo_matrix((values, (users, items)), shape=(self.n_user, self.n_item)).tolil()

        return lil_record