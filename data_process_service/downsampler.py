#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error


class DownSampler(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_downsampler'

    def downsample_user(self, record, n=None, frac=None):

        record = self.downsample_core(record, 'uid', n=n, frac=frac)
        return record

    def downsample_item(self, record, n=None, frac=None):

        record = self.downsample_core(record, 'iid', n=n, frac=frac)
        return record

    def downsample_core(self, record, col, n=None, frac=None):

        sample_col = record[col].drop_duplicates().sample(n=n, frac=frac)

        record = record.merge(sample_col, on=col).reset_index(drop=True)

        return record

