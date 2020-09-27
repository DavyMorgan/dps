#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import pandas as pd
import numpy as np


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
        self.split_core(record, splits)
        self.drop_rank_and_reset_index()

        return self.train_record, self.val_record, self.test_record
    
    def rank(self, record):

        raise NotImplementedError
    
    def split_core(self, record, splits):

        self.train_record = record[record['rank'] > splits[2] + splits[1]]
        
        val_test_record = record[record['rank'] <= splits[2] + splits[1]].copy()
        val_test_record['rank'] = val_test_record.groupby('uid')['rank'].transform(np.random.permutation)

        self.val_record = val_test_record[val_test_record['rank'] > splits[2]]
        self.test_record = val_test_record[val_test_record['rank'] <= splits[2]]
    
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


class RandomSplitter(Splitter):

    def __init__(self, flags_obj, record):

        super(RandomSplitter, self).__init__(flags_obj, record)

    def split(self, record, splits):

        num_record = len(record)
        splits = np.array(splits)
        splits = (num_record*splits).astype(np.int32)

        results = []
        for n in splits[:-1]:
            sampled_record = record.sample(n=n, random_state=np.random.RandomState()).reset_index(drop=True)
            results.append(sampled_record)
            record = pd.concat([record, sampled_record]).drop_duplicates(keep=False).reset_index(drop=True)

        results.append(record)

        return results


class SkewSplitter(PercentageSplitter):

    def __init__(self, flags_obj, record):

        super(SkewSplitter, self).__init__(flags_obj, record)

    def split_3(self, record, splits, cap=None):

        popularity = record[['iid', 'uid']].groupby('iid').count().reset_index().rename(columns={'uid': 'pop'})
        record = record.merge(popularity, on='iid')
        record['pop'] = record['pop'].apply(lambda x : 1/x)

        if cap is not None:
            pop = record['pop'].to_numpy()
            pop = np.unique(pop)
            cap_threshold = np.percentile(pop, cap)
            record['pop'] = record['pop'].apply(lambda x : min(x, cap_threshold))

        self.test_record = record.groupby('uid').apply(pd.DataFrame.sample, frac=splits[2], weights='pop').reset_index(drop=True)

        train_val_record = pd.concat([record, self.test_record]).drop_duplicates(keep=False).reset_index(drop=True)
        train_val_record = self.rank(train_val_record)

        self.train_record = train_val_record[train_val_record['rank'] >= splits[1]/(splits[0]+splits[1])]
        self.val_record = train_val_record[train_val_record['rank'] < splits[1]/(splits[0]+splits[1])]

        self.drop_and_reset_index_3()

        return self.train_record, self.val_record, self.test_record

    def split_2(self, record, splits, cap=None):

        popularity = record[['iid', 'uid']].groupby('iid').count().reset_index().rename(columns={'uid': 'pop'})
        record = record.merge(popularity, on='iid')
        record['pop'] = record['pop'].apply(lambda x : 1/x)

        if cap is not None:
            pop = record['pop'].to_numpy()
            pop = np.unique(pop)
            cap_threshold = np.percentile(pop, cap)
            record['pop'] = record['pop'].apply(lambda x : min(x, cap_threshold))

        self.val_test_record = record.groupby('uid').apply(pd.DataFrame.sample, frac=splits[1], weights='pop').reset_index(drop=True)

        train_record = pd.concat([record, self.val_test_record]).drop_duplicates(keep=False).reset_index(drop=True)

        self.train_record = train_record

        self.drop_and_reset_index_2()

        return self.train_record, self.val_test_record

    def split(self, record, splits, cap=None):

        if len(splits) == 3:
            return self.split_3(record, splits, cap)
        elif len(splits) == 2:
            return self.split_2(record, splits, cap)

    def unbiased_split(self, record, splits, cap=None):

        popularity = record[['iid', 'uid']].groupby('iid').count().reset_index().rename(columns={'uid': 'pop'})
        record = record.merge(popularity, on='iid')
        record['pop'] = record['pop'].apply(lambda x : 1/x)

        if cap is not None:
            pop = record['pop'].to_numpy()
            pop = np.unique(pop)
            cap_threshold = np.percentile(pop, cap)
            record['pop'] = record['pop'].apply(lambda x : min(x, cap_threshold))

        self.val_test_record = record.sample(frac=splits[1], weights='pop', random_state=np.random.RandomState()).reset_index(drop=True)

        train_record = pd.concat([record, self.val_test_record]).drop_duplicates(keep=False).reset_index(drop=True)

        self.train_record = train_record

        self.drop_and_reset_index_2()

        return self.train_record, self.val_test_record

    def drop_and_reset_index_3(self):

        self.train_record = self.train_record.drop(columns=['rank', 'pop']).reset_index(drop=True)
        self.val_record = self.val_record.drop(columns=['rank', 'pop']).reset_index(drop=True)
        self.test_record = self.test_record.drop(columns=['pop']).reset_index(drop=True) 

    def drop_and_reset_index_2(self):

        self.train_record = self.train_record.drop(columns=['pop']).reset_index(drop=True)
        self.val_test_record = self.val_test_record.drop(columns=['pop']).reset_index(drop=True) 


class TemporalSplitter(PercentageSplitter):

    def __init__(self, flags_obj, record):

        self.name = flags_obj.name + '_temporal_splitter' 

    def split_2(self, record, splits):

        record = self.rank(record)

        self.early_record = record[record['rank'] >= splits[1]]
        self.late_record = record[record['rank'] < splits[1]]

        self.drop_rank_and_reset_index_2()

        return self.early_record, self.late_record

    def split_3(self, record, splits):

        record = self.rank(record)
        record_cp = record.copy()
        record_cp['rank'] = record_cp.groupby('uid')['rank'].transform(np.random.permutation)

        self.late_record = record_cp[record_cp['rank'] <= splits[2]]
        self.early_record = record_cp[record_cp['rank'] >= splits[1] + splits[2]]
        self.middle_record = record_cp[(record_cp['rank'] < splits[1] + splits[2]) & (record_cp['rank'] > splits[2])]

        self.drop_rank_and_reset_index_3()

        return self.early_record, self.middle_record, self.late_record

    def split(self, record, splits):

        if len(splits) == 2:
            return self.split_2(record, splits)
        elif len(splits) == 3:
            return self.split_3(record, splits)

    def drop_rank_and_reset_index_2(self):

        self.early_record = self.early_record.drop(columns=['rank']).reset_index(drop=True)
        self.late_record = self.late_record.drop(columns=['rank']).reset_index(drop=True)

    def drop_rank_and_reset_index_3(self):

        self.early_record = self.early_record.drop(columns=['rank']).reset_index(drop=True)
        self.middle_record = self.middle_record.drop(columns=['rank']).reset_index(drop=True)
        self.late_record = self.late_record.drop(columns=['rank']).reset_index(drop=True)
