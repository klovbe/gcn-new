from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import CellGeneGCN, MLP
from gcn.dataset import DataSet

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'baron', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
# flags.DEFINE_string("data_dir", "F:/single_cell_data/baron", "data_dir")
flags.DEFINE_string("data_dir", "./data/baron_random", "data_dir")
flags.DEFINE_string("ppi_path", "./data/baron_random/ppi_dense.csv", "ppi path")
flags.DEFINE_string('model', 'CellGeneGCN', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

print("loading data...")
gene_adj, coo_gene_feature, gene_labels, cell_names_num, cell_set_size, cell_x, cell_y\
 = load_cell_gene_data_variable(FLAGS.data_dir, FLAGS.ppi_path)

gene_names_num = len(gene_labels)  # df
cell_gene_num = cell_x.shape[1]  # df
cell_label_num = cell_y.shape[1]  # df

sparse_gene_features = sparse_to_tuple(coo_gene_feature)  # df

if FLAGS.model == "CellGeneGCN":
    sparse_gene_adj = [preprocess_adj(gene_adj)]  # array
    num_supports = 1
    model_func = CellGeneGCN
else:
    raise ValueError(FLAGS.model)
input_data_adj = []
feed_dict_adj = []
a,b = load_sparse_data('adj_0', sparse_gene_adj[0])
input_data_adj.append(a)
feed_dict_adj.append(b)
input_data_gene_feature, feed_dict_gene_feature = load_sparse_data('gene_feature', sparse_gene_features)
data = [cell_x, cell_y, np.ones((len(cell_y)), dtype=np.float32), FLAGS.dropout, sparse_gene_features[1].shape]
input_data_dense, feed_dict_dense = load_dense_data(cell_names_num, gene_names_num, cell_label_num, data)

index = tf.placeholder(tf.int64)

# Create model
model = model_func(index, input_data_adj, input_data_gene_feature, input_data_dense, gene_names_num,
                   input_dim=cell_names_num, output_dim=cell_label_num, logging=True)

# Initialize session
sess = tf.Session()
sess.run(tf.variables_initializer(
    list(input_data_adj[0].values())), feed_dict=feed_dict_adj[0])
sess.run(tf.variables_initializer(
    list(input_data_gene_feature.values())), feed_dict=feed_dict_gene_feature)
sess.run(tf.variables_initializer(
    list(input_data_dense.values())), feed_dict=feed_dict_dense)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))

train_dataset = DataSet(cell_set_size[0], sum(cell_set_size[1:3]), label=cell_y)
test_dataset = DataSet(0, sum(cell_set_size[0]), label=cell_y)
val_dataset = DataSet(sum(cell_set_size[0:3]), sum(cell_set_size[3]), label=cell_y)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()

    while True:
        a = train_dataset.next()
        ind,_ = a
        if ind is None:
            train_dataset.reset()
            break
        # Training step
        feed_dict = {index: ind}
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    tot_val_loss = 0.
    pred_probas = []

    while True:
        a = test_dataset.next()
        ind, _ = a
        if ind is None:
            test_dataset.reset()
            break
        # Training step
        feed_dict = {index: ind}
        val_loss, probas = sess.run([model.loss, model.probas], feed_dict=feed_dict)
        pred_probas.append(probas)
        tot_val_loss += val_loss * len(ind)
    pred_probas = np.concatenate(pred_probas, axis=0)
    val_acc = accuracy(pred_probas, val_dataset.label)
    avg_val_loss = tot_val_loss / val_dataset.samples

    cost_val.append(avg_val_loss)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost_val[-1]),
          "val_acc=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break
t1 = time.time()
print("Optimization Finished!")
print("total cost time {}".format(t1 - t))


# # Testing
# test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#       "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
tot_test_loss = 0.
pred_probas = []
cost_test = []

while True:
    a = test_dataset.next()
    ind,_ = a
    if ind is None:
        val_dataset.reset()
        break
    feed_dict = {index: ind}
    test_loss, probas = sess.run([model.loss, model.probas], feed_dict=feed_dict)
    pred_probas.append(probas)
    tot_test_loss += test_loss * len(ind)
pred_probas = np.concatenate(pred_probas, axis=0)
test_acc = accuracy(pred_probas, test_dataset.label)
avg_test_loss = tot_test_loss / test_dataset.samples

cost_test.append(avg_test_loss)

# Print results
print("test_acc=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))
