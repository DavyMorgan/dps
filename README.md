# dps
data process service

## Overview
The data process pipeline of recommendation system research usually follows the fashion of load raw data, filter data, reindex data, split train/val/test datasets, save file and also negative sampling for training recommendation models. This repo serves as a general tool for those above data process operations.

## Running examples
data process of a sampled taobao CTR dataset including 1M user-item interactions.
```
cd examples
python taobao_ctr.py
```

## Requirements
```
numpy
pandas
scipy
absl-py
```

## Components
The best entry point to use the following components is a DataFrame with 'uid', 'iid' and 'ts' in its columns.
### loader
CsvLoader: load csv file  
CooLoader: load coo file (sparse matrix in coordidate format)  
JsonLoader: load json file
### filter
CFFilter: k-core filter  
DuplicationFilter: filter duplicated records with the earliest record left
### reindexer
Reindexer: reindex uid and iid, start from 0
### splitter
AbsoluteSplitter: split the dataset with test and validation sample number fixed  
PercentageSplitter: split the dataset proportionally in chronological order  
RandomSplitter: split the dataset randomly  
SkewSplitter: split the dataset into biased and unbiased parts according to related literatures ([PF](https://dawenl.github.io/publications/LiangCB16-causalrec.pdf), [CausE](https://arxiv.org/pdf/1706.07639.pdf), [DICE](https://arxiv.org/pdf/2006.11011.pdf)).  
### generator
CooGenerator: generate sparse matrix in coordinate format  
LilGenerator: generate sparse matrix in lists in list format  
DokGenerator: generate sparse matrix in dictionary of keys format
### transformer
SparseTransformer: perform sparse matrix format transformation from coo to lil and dok
### saver
CsvSaver: save DataFrame to file  
CooSaver: save coo matrix to file  
JsonSaver: save dict to file
### reporter
CsvReporter: report statistics of the data  
### sampler
PointSampler: negative sampling for pointwise optimization such as logloss  
PairSampler: negtive sampling for pairwise optimization such as bprloss

