#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import pandas as pd
import numpy as np

from tqdm import tqdm


class Splitter(object):

    def __init__(self, flags_obj, record):

        self.name = flags_obj.name + '_splitter'
        self.stat(record)
    
    def stat(self, record):

        self.num_users = record['uid'].nunique()
        self.num_items = record['iid'].nunique()
        self.num_records = len(record)
    
    def split(self, record, splits):

        record = self.rank(record)
        print(record.head(10))
        self.split_core(record, splits)
        self.drop_rank_and_reset_index()
    
    def rank(self, record):

        raise NotImplementedError
    
    def split_core(self, record, splits):

        self.test_record = record[record['rank'] <= splits[2]]
        self.val_record = record[(record['rank'] > splits[2]) & (record['rank'] <= splits[2] + splits[1])]
        self.train_record = record[record['rank'] > splits[2] + splits[1]]
    
    def drop_rank_and_reset_index(self):

        self.train_record = self.train_record.drop(columns=['rank']).reset_index(drop=True)
        self.val_record = self.val_record.drop(columns=['rank']).reset_index(drop=True)
        self.test_record = self.test_record.drop(columns=['rank']).reset_index(drop=True)
        

class AbsoluteSplitter(Splitter):

    def __init__(self, flags_obj, record):

        super(AbsoluteSplitter, self).__init__(flags_obj, record)
    
    def rank(self, record):

        record['rank'] = record['ts'].groupby(record['uid']).rank(method='first', ascending=False)

        return record


class PercentageSplitter(Splitter):

    def __init__(self, flags_obj, record):

        super(PercentageSplitter, self).__init__(flags_obj, record)
    
    def rank(self, record):

        record['rank'] = record['ts'].groupby(record['uid']).rank(method='first', pct=True, ascending=False)

        return record
