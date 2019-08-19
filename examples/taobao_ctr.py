#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags

import pandas as pd

import time
import sys

sys.path.append('../')

import data_process_service.loader as LOADER
import data_process_service.filter as FILTER
import data_process_service.reindexer as REINDEXER
import data_process_service.splitter as SPLITTER
import data_process_service.generator as GENERATOR
import data_process_service.saver as SAVER


FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'test', 'Test name.')
flags.DEFINE_bool('test', False, 'Whether in test mode.')
flags.DEFINE_string('scale', '100M', 'Dataset Scale')
flags.DEFINE_string('load_path', '', 'Path to load file.')
flags.DEFINE_string('save_path', './data/taobao_ctr/output', 'Path to save file.')


def load_taobao_ctr(flags_obj):

    if flags_obj.scale == '1M':
        filename = 'taobao_ctr_sampled.csv'
        flags_obj.load_path = './data/taobao_ctr/'
        loader = LOADER.CsvLoader(flags_obj)
        record = loader.load(filename, index_col=0)
    elif flags_obj.scale == '100M':
        filename = 'UserBehavior.csv'
        flags_obj.load_path = '/home/zhengyu/data/taobao/'
        loader = LOADER.CsvLoader(flags_obj)
        record = loader.load(filename, header=None, names=['uid', 'iid', 'cid', 'behavior', 'ts'])

    return record


def filter_duplication_taobao_ctr(flags_obj, record):

    duplication_filter = FILTER.DuplicationFilter(flags_obj, record)
    record = duplication_filter.filter(record)

    return record


def filter_cf_taobao_ctr(flags_obj, record):

    cf_filter = FILTER.CFFilter(flags_obj, record)
    record = cf_filter.filter_item_k_core(record, 10)
    record = cf_filter.filter_user_k_core(record, 10)

    return record


def reindex_taobao_ctr(flags_obj, record):

    reindexer = REINDEXER.Reindexer(flags_obj)
    record = reindexer.reindex_item(record)
    record = reindexer.reindex_user(record)

    return record


def split_taobao_ctr(flags_obj, record):

    splitter = SPLITTER.PercentageSplitter(flags_obj, record)
    splits = [0.6,0.2,0.2]
    splitter.split(record, splits)

    return splitter.train_record, splitter.val_record, splitter.test_record


def generate_coo_taobao_ctr(flags_obj, record, **kwargs):

    generator = GENERATOR.CooGenerator(flags_obj)
    coo_record = generator.generate(record, **kwargs)

    return coo_record


def save_coo_taobao_ctr(flags_obj, coo_record):

    saver = SAVER.CooSaver(flags_obj)
    filename = 'taobao_ctr_train_coo.npz'
    saver.save_file(filename, coo_record)


def test_taobao_ctr(flags_obj):

    start_time = time.time()
    record = load_taobao_ctr(flags_obj)
    load_time = time.time() - start_time
    print('load time: {:.2f} s'.format(load_time))
    print('num record: {}'.format(len(record)))

    start_time = time.time()
    record = filter_duplication_taobao_ctr(flags_obj, record)
    filter_duplication_time = time.time() - start_time
    print('filter duplication time: {:.2f} s'.format(filter_duplication_time))
    print('num record: {}'.format(len(record)))

    start_time = time.time()
    record = filter_cf_taobao_ctr(flags_obj, record)
    filter_cf_time = time.time() - start_time
    print('filter cf time: {:.2f} s'.format(filter_cf_time))
    print('num record: {}'.format(len(record)))

    start_time = time.time()
    record = reindex_taobao_ctr(flags_obj, record)
    reindex_time = time.time() - start_time
    print('reindex time: {:.2f} s'.format(reindex_time))

    start_time = time.time()
    train_record, val_record, test_record = split_taobao_ctr(flags_obj, record)
    split_time = time.time() - start_time
    print('split time: {:.2f} s'.format(split_time))

    start_time = time.time()
    n_user = record['uid'].nunique()
    n_item = record['iid'].nunique()
    coo_record = generate_coo_taobao_ctr(flags_obj, train_record, n_user=n_user, n_item=n_item)
    generate_coo_time = time.time() - start_time
    print('generate coo time: {:.2f} s'.format(generate_coo_time))
    print('num record: {}'.format(coo_record.nnz))

    start_time = time.time()
    save_coo_taobao_ctr(flags_obj, coo_record)
    save_coo_time = time.time() - start_time
    print('save time: {:.2f} s'.format(save_coo_time))


def main(argv):

    flags_obj = flags.FLAGS

    test_taobao_ctr(flags_obj)


if __name__ == "__main__":

    app.run(main)
