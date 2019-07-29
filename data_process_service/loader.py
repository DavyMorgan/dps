#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import pandas as pd

import os


class Loader(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_loader'
        self.load_path = flags_obj.load_path
        self.save_path = flags_obj.save_path
        if not flags_obj.test:
            self.check_save_path()
    
    def check_save_path(self):

        if not os.path.exists(self.save_path):

            os.mkdir(self.save_path)
    
    def load_file(self, filename, argu_dict):

        raise NotImplementedError


class CSVLoader(Loader):

    def __init__(self, flags_obj):

        super(CSVLoader, self).__init__(flags_obj)
    
    def load_file(self, filename, argu_dict):

        argu_dict['filepath_or_buffer'] = filename
        self.record = pd.read_csv(**argu_dict)

