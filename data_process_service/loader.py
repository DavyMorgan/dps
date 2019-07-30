#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import pandas as pd
import scipy.sparse as sp

import os


class Loader(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_loader'
        self.load_path = flags_obj.load_path
        self.check_load_path()
    
    def check_load_path(self):

        if not os.path.exists(self.load_path):

            print('Error! Load path ({}) does not exist!'.format(self.load_path))
    
    def load_file(self, filename, **kwargs):

        raise NotImplementedError


class CSVLoader(Loader):

    def __init__(self, flags_obj):

        super(CSVLoader, self).__init__(flags_obj)
    
    def load_file(self, filename, **kwargs):

        filename = os.path.join(self.load_path, filename)
        self.record = pd.read_csv(filename, **kwargs)


class COOLoader(Loader):

    def __init__(self, flags_obj):

        super(COOLoader, self).__init__(flags_obj)
    
    def load_file(self, filename, **kwargs):

        filename = os.path.join(self.load_path, filename)
        self.record = sp.load_npz(filename)
