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
import splitter as SPLITTER
import generator as GENERATOR


FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'test', 'Test name.')
flags.DEFINE_bool('test', True, 'Whether in test mode.')
flags.DEFINE_string('load_path', '/home/zhengyu/data/taobao', 'Path to load file.')
flags.DEFINE_string('save_path', '/home/zhengyu/data/taobao', 'Path to save file.')


def test_csvloader(flags_obj):

    loader = LOADER.CSVLoader(flags_obj)
    filename = 'UserBehavior.csv'
    loader.load_file(filename, header=None, names=['uid', 'iid', 'cid', 'behavior', 'ts'])
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


def test_absolutesplitter(flags_obj):

    #record = pd.DataFrame({'uid': [3,3,3,4,4,4,4], 'iid': [1,2,3,4,5,6,7], 'ts': [1,2,3,4,5,6,7]})
    record = pd.DataFrame({'uid': [4,4,4,4,2,2], 'iid': [10,9,8,7,6,5], 'ts': [6,5,4,3,2,1]})
    splitter = SPLITTER.AbsoluteSplitter(flags_obj, record)
    print(record.head(10))
    print('data frames before split!')

    record = record.sample(frac=1.0)
    splits = [-1, 1, 1]
    splitter.split(record, splits)

    print(splitter.train_record.head(10))
    print('train split!')

    print(splitter.val_record.head(10))
    print('val split!')

    print(splitter.test_record.head(10))
    print('test split!')


def test_percentagesplitter(flags_obj):

    record = pd.DataFrame({'uid': [2,2,4,4,4,4], 'iid': [10,9,8,7,6,5], 'ts': [6,5,4,3,2,1]})
    #record = pd.DataFrame({'uid': [4,4,4,4,4,2,2], 'iid': [10,9,8,7,6,5,4], 'ts': [6,6,6,6,6,6,6]})
    splitter = SPLITTER.PercentageSplitter(flags_obj, record)
    print(record.head(10))
    print('data frames before split!')

    record = record.sample(frac=1.0)
    splits = [0.6,0.2,0.2]
    splitter.split(record, splits)

    print(splitter.train_record.head(10))
    print('train split!')

    print(splitter.val_record.head(10))
    print('val split!')

    print(splitter.test_record.head(10))
    print('test split!')


def test_coogenerator(flags_obj):

    record = pd.DataFrame({'uid': [0,0,0,1,1,1,1], 'iid': [0,1,2,3,4,5,6], 'ts': [1,2,3,4,5,6,7]})
    generator = GENERATOR.CooGenerator(flags_obj)

    print(record.head(10))
    print('original record!')

    coo_record = generator.generate(record)

    print('row: \t{}'.format(coo_record.row))
    print('col: \t{}'.format(coo_record.col))
    print('coo record!')


def test_lilgenerator(flags_obj):

    record = pd.DataFrame({'uid': [0,0,0,1,1,1,1], 'iid': [0,1,2,3,4,5,6], 'ts': [1,2,3,4,5,6,7]})
    generator = GENERATOR.LilGenerator(flags_obj)

    print(record.head(10))
    print('original record!')

    lil_record = generator.generate(record)

    print('user 0: {}'.format(lil_record.rows[0]))
    print('user 1: {}'.format(lil_record.rows[1]))
    print('lil record!')


def test_dokgenerator(flags_obj):

    record = pd.DataFrame({'uid': [0,0,0,1,1,1,1], 'iid': [0,1,2,3,4,5,6], 'ts': [1,2,3,4,5,6,7]})
    generator = GENERATOR.DokGenerator(flags_obj)

    print(record.head(10))
    print('original record!')

    dok_record = generator.generate(record)

    for (u, i) in dok_record.keys():

        print('{} {}'.format(u, i ))
    
    print('dok record!')


def main(argv):

    flags_obj = flags.FLAGS

    test_csvloader(flags_obj)
    #test_cffilter(flags_obj)
    #test_reindexer(flags_obj)
    #test_absolutesplitter(flags_obj)
    #test_percentagesplitter(flags_obj)
    #test_coogenerator(flags_obj)
    #test_lilgenerator(flags_obj)
    #test_dokgenerator(flags_obj)


if __name__ == "__main__":

    app.run(main)
