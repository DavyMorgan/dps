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

    def generate(self, record, **kwargs):

        raise NotImplementedError

    def stat(self, record, **kwargs):

        self.n_record = len(record)

        if 'n_user' in kwargs:
            self.n_user = kwargs['n_user']
        else:
            self.n_user = record['uid'].nunique()

        if 'n_item' in kwargs:
            self.n_item = kwargs['n_item']
        else:
            self.n_item = record['iid'].nunique()


class SparseGenerator(Generator):

    def __init__(self, flags_obj):

        super(SparseGenerator, self).__init__(flags_obj)

    def generate(self, record, **kwargs):

        raise NotImplementedError

    def generate_coo(self, record, **kwargs):

        self.stat(record, **kwargs)

        values = np.ones(self.n_record)
        users = record['uid'].to_numpy()
        items = record['iid'].to_numpy()
        coo_record = sp.coo_matrix((values, (users, items)), shape=(self.n_user, self.n_item))

        return coo_record


class CooGenerator(SparseGenerator):

    def __init__(self, flags_obj):

        super(CooGenerator, self).__init__(flags_obj)

    def generate(self, record, **kwargs):

        coo_record = self.generate_coo(record, **kwargs)

        return coo_record


class LilGenerator(SparseGenerator):

    def __init__(self, flags_obj):

        super(LilGenerator, self).__init__(flags_obj)

    def generate(self, record, **kwargs):

        lil_record = self.generate_coo(record, **kwargs).tolil()

        return lil_record


class DokGenerator(SparseGenerator):

    def __init__(self, flags_obj):

        super(DokGenerator, self).__init__(flags_obj)

    def generate(self, record, **kwargs):

        dok_record = self.generate_coo(record, **kwargs).todok()

        return dok_record
