#coding=utf-8

import os
import sys
sys.path.append('/home/zhengyu/workspace/dps')
import time

from absl import app
from absl import flags

import numpy as np
import pandas as pd

import data_process_service.utils as utils

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'taobao_preprocess', 'Test name.')
flags.DEFINE_bool('test', False, 'Whether in test mode.')
flags.DEFINE_string('load_path', './data/taobao_ctr/', 'Path to load file.')
flags.DEFINE_string('save_path', './data/taobao_ctr/output', 'Path to save file.')


def filter_items_with_multiple_cids_taobao_ctr(flags_obj, record):

    start_time = time.time()
    item_cate = record[['iid', 'cid']].drop_duplicates().groupby('iid').count().reset_index().rename(columns={'cid': 'count'})
    items_with_single_cid = item_cate[item_cate['count'] == 1]['iid']

    id_filter = utils.FILTER.IDFilter(flags_obj, record)
    record = id_filter.filter(record, 'iid', items_with_single_cid)
    print('filter items with multiple_cids time: {:.2f} s'.format(time.time() - start_time))

    return record


def filter_behavior(flags_obj, record):

    start_time = time.time()
    click = pd.DataFrame({'behavior': ['pv']})
    id_filter = utils.FILTER.IDFilter(flags_obj, record)
    record = id_filter.filter(record, 'behavior', click)
    print('filter behavior time: {:.2f} s'.format(time.time() - start_time))

    return record


def process_taobao(flags_obj):

    record = utils.load_csv(flags_obj, 'taobao_ctr_sampled.csv', index_col=0)
    record = utils.filter_duplication(flags_obj, record)
    record = filter_items_with_multiple_cids_taobao_ctr(flags_obj, record)
    record = filter_behavior(flags_obj, record)
    record = utils.downsample_user(flags_obj, record, 0.5)
    record = utils.filter_cf(flags_obj, record, 10)
    record, user_reindex_map, item_reindex_map = utils.reindex_user_item(flags_obj, record)
    record, cate_reindex_map = utils.reindex_feature(flags_obj, record, 'cid')
    utils.save_reindex_user_item_map(flags_obj, user_reindex_map, item_reindex_map)
    utils.save_reindex_feature_map(flags_obj, 'cate', cate_reindex_map)
    train_record, val_record, test_record = utils.split(flags_obj, record, [0.6, 0.2, 0.2])
    utils.save_csv_record(flags_obj, record, train_record, val_record, test_record)
    utils.report(flags_obj, record)
    utils.extract_save_item_feature(flags_obj, record, 'cate', 'cid')
    coo_record, train_coo_record, val_coo_record, test_coo_record = utils.generate_coo(flags_obj, record, train_record, val_record, test_record)
    utils.save_coo(flags_obj, coo_record, train_coo_record, val_coo_record, test_coo_record)
    utils.compute_popularity(flags_obj, train_coo_record)
    utils.compute_popularity(flags_obj, coo_record, 'popularity_all.npy')
    utils.generate_graph(flags_obj, train_coo_record)


def main(argv):

    flags_obj = flags.FLAGS

    process_taobao(flags_obj)


if __name__ == "__main__":

    app.run(main)

