#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import pandas as pd
import scipy.sparse as sp

import json

import os


class Saver(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_saver'
        self.save_path = flags_obj.save_path
        if not flags_obj.test:
            self.check_save_path()
    
    def check_save_path(self):

        if not os.path.exists(self.save_path):

            os.mkdir(self.save_path)
    
    def save(self, filename, data):

        raise NotImplementedError


class CsvSaver(Saver):

    def __init__(self, flags_obj):

        super(CsvSaver, self).__init__(flags_obj)
    
    def save(self, filename, data):

        filename = os.path.join(self.save_path, filename)
        data.to_csv(filename)


class CooSaver(Saver):

    def __init__(self, flags_obj):

        super(CooSaver, self).__init__(flags_obj)
    
    def save(self, filename, data):

        filename = os.path.join(self.save_path, filename)
        sp.save_npz(filename, data)


class JsonSaver(Saver):

    def __init__(self, flags_obj):

        super(JsonSaver, self).__init__(flags_obj)
    
    def save(self, filename, data):

        filename = os.path.join(self.save_path, filename)
        with open(filename, 'w') as f:
            f.write(json.dumps(data))


class NpySaver(Saver):

    def __init__(self, flags_obj):

        super(NpySaver, self).__init__(flags_obj)
    
    def save(self, filename, data):

        filename = os.path.join(self.save_path, filename)
        np.save(filename, data)
