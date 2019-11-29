#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8

import numpy as np
import scipy.sparse as sp


class Grapher(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_grapher'

    def generate_coo_adj_graph(self, coo_record):

        num_record = coo_record.nnz
        num_user = coo_record.shape[0]
        num_item = coo_record.shape[1]

        values = np.ones(2*num_record)
        row = coo_record.row
        col = coo_record.col

        col = col + num_user

        bi_row = np.hstack([row, col])
        bi_col = np.hstack([col, row])

        coo_adj_graph = sp.coo_matrix((values, (bi_row, bi_col)), shape=(num_user+num_item, num_user+num_item))

        return coo_adj_graph

