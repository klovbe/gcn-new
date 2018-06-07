#encoding=utf-8

__author__ = "turingli(liyi)"

import os
import sys
import codecs
import numpy as np
import pandas as pd

class DataSet:

  def __init__(self, begin, data, label=None, columns=None, batch_size=128, shuffle=True, nrows=None, dataset_name="train"):

    # data = pd.read_csv(path, sep=",", nrows=nrows)
    # self.columns = list(data.columns)
    # data = np.float32(data.values)
    # self.path = path
    self.dataset_name = dataset_name
    self.columns = columns
    self.samples = data
    self.begin = begin
    self.index = np.array(range(self.samples)) + self.begin
    self.label = label


    self.cur_pos = 0
    self.batch_counter = 0

    self.batch_size = batch_size
    self.shuffle = shuffle

    if shuffle:
      self.shuffle_index()


    print("batch_size is {}, have {} samples, step nums is {}".format(batch_size, self.samples,
                                                                                    self.steps))
    print("make dataset {} end".format(dataset_name))

  def next(self):

    batch_size = self.batch_size

    if self.cur_pos >= self.samples:  # for infer mode
      return None,None

    be, en = self.cur_pos, min(self.samples, self.cur_pos + batch_size)
    batch_index = self.index[be:en]
    batch_label = None
    if self.label is not None:
      batch_label = self.label[batch_index]
    self.cur_pos = en
    self.batch_counter += 1
    # print("getting {}th batch end".format(self.batch_counter))
    return batch_index, batch_label

  def sample_batch(self):
    index = np.random.choice(np.array(range(self.samples)) + self.begin, self.batch_size)
    return index

  def shuffle_index(self):
    self.index = np.random.permutation(np.array(range(self.samples)) + self.begin)

  def reset(self):
    self.cur_pos = 0
    self.batch_counter = 0
    self.index = np.array(range(self.samples)) + self.begin
    if self.shuffle:
      self.shuffle_index()

  @property
  def steps(self):
    return self.samples // self.batch_size
  @property
  def mode(self):
    return self.samples % self.batch_size


if __name__ == "__main__":

  # dataset = DataSet()
  print("xx")