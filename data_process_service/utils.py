#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import scipy.sparse as sp

import time
import sys

sys.path.append('../')

import data_process_service.loader as LOADER
import data_process_service.downsampler as DOWNSAMPLER
import data_process_service.filter as FILTER
import data_process_service.reindexer as REINDEXER
import data_process_service.reporter as REPORTER
import data_process_service.splitter as SPLITTER
import data_process_service.generator as GENERATOR
import data_process_service.saver as SAVER
import data_process_service.grapher as GRAPHER


def load_csv(flags_obj, filename, **kwargs):

    start_time = time.time()

    loader = LOADER.CsvLoader(flags_obj)
    record = loader.load(filename, **kwargs)

    load_time = time.time() - start_time
    print('load time: {:.2f} s'.format(load_time))

    return record


def load_coo(flags_obj, filename, **kwargs):

    start_time = time.time()

    loader = LOADER.CooLoader(flags_obj)
    record = loader.load(filename, **kwargs)

    load_time = time.time() - start_time
    print('load time: {:.2f} s'.format(load_time))

    return record


def filter_duplication(flags_obj, record):

    start_time = time.time()

    duplication_filter = FILTER.DuplicationFilter(flags_obj, record)
    record = duplication_filter.filter(record)

    filter_duplication_time = time.time() - start_time
    print('filter duplication time: {:.2f} s'.format(filter_duplication_time))

    return record


def downsample_user(flags_obj, record, frac):

    start_time = time.time()

    downsampler = DOWNSAMPLER.DownSampler(flags_obj)
    record = downsampler.downsample_user(record, frac=frac)

    downsample_user_time = time.time() - start_time
    print('downsample user time: {:.2f} s'.format(downsample_user_time))

    return record


def downsample_item(flags_obj, record, frac):

    start_time = time.time()

    downsampler = DOWNSAMPLER.DownSampler(flags_obj)
    record = downsampler.downsample_item(record, frac=frac)

    downsample_item_time = time.time() - start_time
    print('downsample item time: {:.2f} s'.format(downsample_item_time))

    return record


def filter_cf(flags_obj, record, k_core):

    start_time = time.time()

    cf_filter = FILTER.CFFilter(flags_obj, record)
    record = cf_filter.filter_item_k_core(record, k_core)
    record = cf_filter.filter_user_k_core(record, k_core)

    filter_cf_time = time.time() - start_time
    print('filter cf time: {:.2f} s'.format(filter_cf_time))

    return record


def reindex_user_item(flags_obj, record):

    start_time = time.time()

    reindexer = REINDEXER.Reindexer(flags_obj)
    record, user_reindex_map = reindexer.reindex_user(record)
    record, item_reindex_map = reindexer.reindex_item(record)

    reindex_user_item_time = time.time() - start_time
    print('reindex user item time: {:.2f} s'.format(reindex_user_item_time))

    return record, user_reindex_map, item_reindex_map


def reindex_feature(flags_obj, record, feature):

    start_time = time.time()

    reindexer = REINDEXER.Reindexer(flags_obj)
    record, feature_reindex_map = reindexer.reindex(record, feature)

    reindex_feature_time = time.time() - start_time
    print('reindex feature time: {:.2f} s'.format(reindex_feature_time))

    return record, feature_reindex_map


def save_reindex_user_item_map(flags_obj, user_reindex_map, item_reindex_map):

    start_time = time.time()

    saver = SAVER.JsonSaver(flags_obj)

    filename = 'user_reindex.json'
    saver.save(filename, user_reindex_map)

    filename = 'item_reindex.json'
    saver.save(filename, item_reindex_map)

    save_reindex_user_item_map_time = time.time() - start_time
    print('save reindex user item map time: {:.2f} s'.format(save_reindex_user_item_map_time))


def save_reindex_feature_map(flags_obj, feature, feature_reindex_map):

    if not isinstance(feature, (list, tuple)):
        feature = [feature]
    if not isinstance(feature_reindex_map, (list, tuple)):
        feature_reindex_map = [feature_reindex_map]

    start_time = time.time()

    saver = SAVER.JsonSaver(flags_obj)

    for f, fm in zip(feature, feature_reindex_map):

        filename = '{}_reindex.json'.format(f)
        saver.save(filename, fm)

    save_reindex_feature_map_time = time.time() - start_time
    print('save reindex {} feature map time: {:.2f} s'.format(feature, save_reindex_feature_map_time))


def split(flags_obj, record, splits):

    start_time = time.time()

    splitter = SPLITTER.PercentageSplitter(flags_obj, record)
    train_record, val_record, test_record = splitter.split(record, splits)

    split_time = time.time() - start_time
    print('split time: {:.2f} s'.format(split_time))

    return train_record, val_record, test_record


def skew_split(flags_obj, record, splits, cap=None):

    start_time = time.time()

    splitter = SPLITTER.SkewSplitter(flags_obj, record)
    train_record, val_record, test_record = splitter.split(record, splits, cap)

    split_time = time.time() - start_time
    print('split time: {:.2f} s'.format(split_time))

    return train_record, val_record, test_record


def skew_split_v2(flags_obj, record, splits, cap=None):

    start_time = time.time()

    splitter = SPLITTER.SkewSplitter(flags_obj, record)
    train_record, val_test_record = splitter.split(record, splits, cap)

    split_time = time.time() - start_time
    print('split time: {:.2f} s'.format(split_time))

    return train_record, val_test_record


def skew_split_v3(flags_obj, record, splits, cap=None):

    start_time = time.time()

    splitter = SPLITTER.SkewSplitter(flags_obj, record)
    train_record, val_test_record = splitter.unbiased_split(record, splits, cap)

    split_time = time.time() - start_time
    print('split time: {:.2f} s'.format(split_time))

    return train_record, val_test_record


def skew_extract(flags_obj, skew_record, frac):

    start_time = time.time()

    splitter = SPLITTER.TemporalSplitter(flags_obj, skew_record)
    skew_train_record, skew_test_record = splitter.split(skew_record, [frac, 1-frac])

    split_time = time.time() - start_time
    print('split time: {:.2f} s'.format(split_time))

    return skew_train_record, skew_test_record


def skew_extract_v2(flags_obj, skew_record, frac):

    start_time = time.time()

    splitter = SPLITTER.TemporalSplitter(flags_obj, skew_record)
    skew_train_record, skew_val_record, skew_test_record = splitter.split(skew_record, frac)

    split_time = time.time() - start_time
    print('split time: {:.2f} s'.format(split_time))

    return skew_train_record, skew_val_record, skew_test_record


def skew_extract_v3(flags_obj, skew_record, frac):

    start_time = time.time()

    splitter = SPLITTER.RandomSplitter(flags_obj, skew_record)
    skew_train_record, skew_val_record, skew_test_record = splitter.split(skew_record, frac)

    split_time = time.time() - start_time
    print('split time: {:.2f} s'.format(split_time))

    return skew_train_record, skew_val_record, skew_test_record


def save_csv_record(flags_obj, record, train_record, val_record, test_record, train_skew_record=None):

    start_time = time.time()

    saver = SAVER.CsvSaver(flags_obj)

    filename = 'record.csv'
    saver.save(filename, record)

    filename = 'train_record.csv'
    saver.save(filename, train_record)

    filename = 'val_record.csv'
    saver.save(filename, val_record)

    filename = 'test_record.csv'
    saver.save(filename, test_record)

    if isinstance(train_skew_record, pd.DataFrame):
        filename = 'train_skew_record.csv'
        saver.save(filename, train_skew_record)

    save_csv_time = time.time() - start_time
    print('save csv time: {:.2f} s'.format(save_csv_time))


def report(flags_obj, record):

    start_time = time.time()

    reporter = REPORTER.CsvReporter(flags_obj)
    reporter.report(record)

    report_time = time.time() - start_time
    print('report time: {:.2f} s'.format(report_time))


def extract_save_item_feature(flags_obj, record, feature, col):

    start_time = time.time()

    item_feature = record[['iid', col]].drop_duplicates().reset_index(drop=True)
    saver = SAVER.CsvSaver(flags_obj)

    filename = 'item_{}_feature.csv'.format(feature)
    saver.save(filename, item_feature)

    extract_save_item_feature_time = time.time() - start_time
    print('extract save item feature time: {:.2f} s'.format(extract_save_item_feature_time))


def generate_coo(flags_obj, record, train_record, val_record, test_record, train_skew_record=None):

    start_time = time.time()

    n_user = record['uid'].nunique()
    n_item = record['iid'].nunique()

    generator = GENERATOR.CooGenerator(flags_obj)
    coo_record = generator.generate(record, n_user=n_user, n_item=n_item)
    train_coo_record = generator.generate(train_record, n_user=n_user, n_item=n_item)
    val_coo_record = generator.generate(val_record, n_user=n_user, n_item=n_item)
    test_coo_record = generator.generate(test_record, n_user=n_user, n_item=n_item)
    if isinstance(train_skew_record, pd.DataFrame):
        train_skew_coo_record = generator.generate(train_skew_record, n_user=n_user, n_item=n_item)

    generate_coo_time = time.time() - start_time
    print('generate coo time: {:.2f} s'.format(generate_coo_time))

    if isinstance(train_skew_record, pd.DataFrame):
        return coo_record, train_coo_record, val_coo_record, test_coo_record, train_skew_coo_record
    else:
        return coo_record, train_coo_record, val_coo_record, test_coo_record


def save_coo(flags_obj, coo_record, train_coo_record, val_coo_record, test_coo_record, train_skew_coo_record=None):

    start_time = time.time()

    saver = SAVER.CooSaver(flags_obj)

    filename = 'coo_record.npz'
    saver.save(filename, coo_record)

    filename = 'train_coo_record.npz'
    saver.save(filename, train_coo_record)

    filename = 'val_coo_record.npz'
    saver.save(filename, val_coo_record)

    filename = 'test_coo_record.npz'
    saver.save(filename, test_coo_record)

    if isinstance(train_skew_coo_record, sp.coo_matrix):
        filename = 'train_skew_coo_record.npz'
        saver.save(filename, train_skew_coo_record)

    save_coo_time = time.time() - start_time
    print('save coo time: {:.2f} s'.format(save_coo_time))


def compute_popularity(flags_obj, coo_record, filename=None):

    start_time = time.time()

    popularity = np.zeros(coo_record.shape[1], dtype=np.int64)
    dok_record = coo_record.todok()
    df = pd.DataFrame(list(dok_record.keys()), columns=['uid', 'iid'])
    df = df.groupby('iid').count().reset_index().rename(columns={'uid': 'count'})
    popularity[df['iid']] = df['count']

    if not filename:
        filename = 'popularity.npy'
    saver = SAVER.NpySaver(flags_obj)
    saver.save(filename, popularity)

    compute_time = time.time() - start_time
    print('compute and save popularity time: {:.2f} s'.format(compute_time))


def generate_graph(flags_obj, train_coo_record, filename='train_coo_adj_graph.npz'):

    start_time = time.time()

    grapher = GRAPHER.Grapher(flags_obj)
    train_coo_adj_graph = grapher.generate_coo_adj_graph(train_coo_record)

    saver =SAVER.CooSaver(flags_obj)
    saver.save(filename, train_coo_adj_graph)

    generate_time = time.time() - start_time
    print('generate adj time: {:.2f} s'.format(generate_time))

