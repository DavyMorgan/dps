#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import pandas as pd


class Reindexer(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_reindexer'
    
    def reindex_core(self, record, column_name):

        keys = record[column_name].unique()
        num_entities = record[column_name].nunique()
        reindex_map = {keys[i]: i for i in range(num_entities)}

        record[column_name] = record[column_name].map(reindex_map)

        return record, reindex_map
    
    def reindex_user(self, record):

        record, reindex_map = self.reindex_core(record, 'uid')
        return record, reindex_map
    
    def reindex_item(self, record):

        record, reindex_map = self.reindex_core(record, 'iid')
        return record, reindex_map