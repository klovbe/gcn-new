from __future__ import division
from __future__ import print_function

import numpy as np

from gcn_chi.utils import *
from gcn_chi.dataset import DataSet
from gcn_chi.metrics import *

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import sklearn.metrics

import time
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "./data/baron_random", "data_dir")
flags.DEFINE_string("ppi_path", "./data/baron_random/ppi_dense.csv", "ppi path")
flags.DEFINE_bool('permutation', False, 'to decide whether to permutation all data among batches.')

gene_adj, coo_gene_feature, gene_labels, cell_names_num, cell_set_size, cell_x, cell_y\
 = load_cell_gene_data_variable(FLAGS.data_dir, FLAGS.ppi_path)

if FLAGS.permutation:
    index_p = np.random.permutation(list(range(len(cell_x))))
    cell_x = cell_x[index_p]
    cell_y = cell_y[index_p]


X = cell_x[cell_set_size[0]:sum(cell_set_size[0:3])]
Y = cell_y[cell_set_size[0]:sum(cell_set_size[0:3])]
Y = np.argmax(Y, 1)
X_test = cell_x[0:cell_set_size[0]]
Y_test = cell_y[0:cell_set_size[0]]
Y_test = np.argmax(Y_test, 1)

t = time.time()
clf = svm.SVC()
clf.fit(X, Y)
Y_pred = clf.predict(X_test)
acc = np.mean(np.equal(Y_test, Y_pred))
kappa = sklearn.metrics.cohen_kappa_score(Y_test, Y_pred)
print("SVM:test_acc=", "{:.5f}".format(acc), "test_kappa=", "{:.5f}".format(kappa),
      "time=", "{:.5f}".format(time.time() - t))


rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(X, Y)
Y_pred = rf.predict(X_test)
acc = np.mean(np.equal(Y_test, Y_pred))
kappa = sklearn.metrics.cohen_kappa_score(Y_test, Y_pred)
print("RandomForest:test_acc=", "{:.5f}".format(acc), "test_kappa=", "{:.5f}".format(kappa),
      "time=", "{:.5f}".format(time.time() - t))

rf = AdaBoostClassifier()
rf.fit(X, Y)
Y_pred = rf.predict(X_test)
acc = np.mean(np.equal(Y_test, Y_pred))
kappa = sklearn.metrics.cohen_kappa_score(Y_test, Y_pred)
print("AdaBoost:test_acc=", "{:.5f}".format(acc), "test_kappa=", "{:.5f}".format(kappa),
      "time=", "{:.5f}".format(time.time() - t))