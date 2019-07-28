#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags

import pandas as pd

import loader as LOADER
import rek_filter as FILTER
import reindexer as REINDEXER


FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'test', 'Test name.')
flags.DEFINE_bool('test', True, 'Whether in test mode.')
flags.DEFINE_string('load_path', '', 'Path to load file.')
flags.DEFINE_string('save_path', '', 'Path to save file.')


def test_csvloader(flags_obj):

    loader = LOADER.CSVLoader(flags_obj)
    filename = '/home/zhengyu/data/taobao/UserBehavior.csv'
    argu_dict = {
        'header': None,
        'names': ['uid', 'iid', 'cid', 'behavior', 'ts']
    }
    loader.load_file(filename, argu_dict)
    loader.record.info()

    return loader.record


def test_cffilter(flags_obj):

    record = pd.DataFrame({'uid': [1,2,2,3,3,3,4,4,4,4], 'iid': [1,2,2,3,3,3,4,4,4,4]})
    rek_filter = FILTER.CFFilter(flags_obj, record)
    print(record.head(10))
    print('{} records before filter!'.format(len(record)))

    record = rek_filter.filter_user_k_core(record, 2)
    print(record.head(10))
    print('{} records after filter!'.format(len(record)))


def test_reindexer(flags_obj):

    record = pd.DataFrame({'uid': [2,2,4,4,4,4], 'iid': [5,6,7,8,9,10]})
    reindexer = REINDEXER.Reindexer(flags_obj)
    print(record.head(10))
    print('data frames before reindex!')

    reindexer.reindex_user(record)
    reindexer.reindex_item(record)

    print(record.head(10))
    print('data frames after reindex!')


def main(argv):

    flags_obj = flags.FLAGS

    #test_csvloader(flags_obj)
    #test_cffilter(flags_obj)
    test_reindexer(flags_obj)


if __name__ == "__main__":

    app.run(main)