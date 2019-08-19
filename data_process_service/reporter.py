#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Reporter(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_reporter'

    def report(self, record):

        raise NotImplementedError


class CsvReporter(Reporter):

    def __init__(self, flags_obj):

        super(CsvReporter, self).__init__(flags_obj)
    
    def report(self, record):

        self.compute(record)

        self.show()
    
    def compute(self, record):

        self.compute_general(record)
        self.compute_user_group(record)
        self.compute_item_group(record)
    
    def compute_general(self, record):

        self.n_user = record['uid'].nunique()
        self.n_item = record['iid'].nunique()
        self.n_record = len(record)
        self.sparsity = self.n_record / self.n_user / self.n_item
    
    def compute_group_core(self, record, group_name):

        record_group = record \
                       .groupby('entity') \
                       .count() \
                       .reset_index()
        
        group_info = record_group['count'].describe()

        setattr(self, group_name, group_info)
    
    def compute_user_group(self, record):

        record_user_item = record[['uid', 'iid']].rename(columns={'uid': 'entity', 'iid': 'count'})
        self.compute_group_core(record_user_item, 'user_group')
    
    def compute_item_group(self, record):

        record_item_user = record[['iid', 'uid']].rename(columns={'iid': 'entity', 'uid': 'count'})
        self.compute_group_core(record_item_user, 'item_group')
    
    def show(self):

        self.show_general()
        self.show_user_group()
        self.show_item_group()
    
    def show_general(self):

        self.show_split_line()

        print('general statistics')
        print('number of users: \t{}'.format(self.n_user))
        print('number of items: \t{}'.format(self.n_item))
        print('number of records: \t{}'.format(self.n_record))
        print('sparsity \t\t{}'.format(self.sparsity))
    
    def show_group_core(self, group_name):

        group_info = getattr(self, group_name)

        self.show_split_line()
        print(group_name)
        print(group_info.to_string())
    
    def show_user_group(self):

        self.show_group_core('user_group')
    
    def show_item_group(self):

        self.show_group_core('item_group')
    
    def show_split_line(self):

        print(''.join(['*' for _ in range(81)]))
