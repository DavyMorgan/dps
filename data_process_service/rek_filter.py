#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import pandas as pd


class Filter(object):

    def __init__(self, flags_obj, record):

        self.name = flags_obj.name + '_filter'
        self.check_columns(record)
    
    def check_columns(self, record):

        raise NotImplementedError


class CFFilter(Filter):

    def __init__(self, flags_obj, record):

        super(CFFilter, self).__init__(flags_obj, record)
    
    def check_columns(self, record):

        if 'uid' not in record.columns or 'iid' not in record.columns:

            print('columns must contain uid and iid!') 
    
    def filter_k_core(self, record, k_core, filtered_column, count_column):

        stat = record[[filtered_column, count_column]] \
               .groupby(filtered_column) \
               .count() \
               .reset_index() \
               .rename(index=str, columns={count_column: 'count'})
        
        stat = stat[stat['count'] > k_core]

        record = record.merge(stat, on=filtered_column)
        record = record.drop(columns=['count'])

        return record
    
    def filter_user_k_core(self, record, k_core):

        return self.filter_k_core(record, k_core, 'uid', 'iid')
    
    def filter_item_k_core(self, record, k_core):

        return self.filter_k_core(record, k_core, 'iid', 'uid')