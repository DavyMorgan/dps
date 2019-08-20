#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import pandas as pd

import sys


class Filter(object):

    def __init__(self, flags_obj, record):

        self.name = flags_obj.name + '_filter'
        if not self.check(record):
            print('DataFrame column check failed!')
            sys.exit()
    
    def check(self, record):

        raise NotImplementedError
    
    def check_columns(self, record, columns):

        for column in columns:

            if column not in record.columns:

                return False
        
        return True
    
    def check_column(self, record, column):

        return column in record.columns


class CFFilter(Filter):

    def __init__(self, flags_obj, record):

        super(CFFilter, self).__init__(flags_obj, record)
    
    def check(self, record):

        return self.check_columns(record, ['uid', 'iid'])
    
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


class DuplicationFilter(Filter):

    def __init__(self, flags_obj, record):

        super(DuplicationFilter, self).__init__(flags_obj, record)
    
    def check(self, record):

        return self.check_columns(record, ['uid', 'iid', 'ts'])
    
    def filter(self, record):

        record = record.sort_values('ts').drop_duplicates(['uid', 'iid']).reset_index(drop=True)

        return record


class IDFilter(Filter):

    def __init__(self, flags_obj, record):

        super(IDFilter, self).__init__(flags_obj, record)
    
    def check(self, record):

        return True
    
    def filter(self, record, column, filtered_ids):

        record = record.merge(filtered_ids, on=column)

        return record
