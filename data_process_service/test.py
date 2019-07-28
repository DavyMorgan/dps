#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags

import loader as LOADER


FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'test', 'Test name.')
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


def main(argv):

    flags_obj = flags.FLAGS

    test_csvloader(flags_obj)


if __name__ == "__main__":

    app.run(main)