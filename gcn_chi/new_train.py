from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn_chi.utils import *
from gcn_chi.models import CellGeneGCN, MLP
from gcn_chi.dataset import DataSet
from gcn_chi.metrics import *

import sklearn.metrics

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
flags.DEFINE_string("save_dir", "./out", "save dir[out_{run_name}]")
flags.DEFINE_string("run_name", "t", "run_name")
flags.DEFINE_string('model', 'CellGeneGCN', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-5, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 400000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('batch_size', 256, 'batch size.')
flags.DEFINE_bool('permutation', False, 'to decide whether to permutation all data among batches.')

out_dir = os.path.join(FLAGS.save_dir, FLAGS.run_name)
if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

print("loading data...")
gene_adj, coo_gene_feature, gene_labels, cell_names_num, cell_set_size, cell_x, cell_y, cell_labels\
 = load_cell_gene_data_variable(FLAGS.data_dir, FLAGS.ppi_path)

if FLAGS.permutation:
    index_p = np.random.permutation(list(range(len(cell_x))))
    cell_x = cell_x[index_p]
    cell_y = cell_y[index_p]

train_dataset = DataSet(cell_set_size[0], sum(cell_set_size[1:3]), label=cell_y, batch_size=FLAGS.batch_size)
val_dataset = DataSet(sum(cell_set_size[0:3]), cell_set_size[3], label=cell_y, batch_size=cell_set_size[3], shuffle=False)
test_dataset = DataSet(0, cell_set_size[0], label=cell_y, batch_size=cell_set_size[0], shuffle=False)


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
data = [cell_x, cell_y, np.ones((len(cell_y)), dtype=np.float32), sparse_gene_features[1].shape]
input_data_dense, feed_dict_dense = load_dense_data(cell_names_num, gene_names_num, cell_label_num, data)

index = tf.placeholder(tf.int64)
dropout_rate = tf.placeholder_with_default(0., shape=[])

# Create model
model = model_func(index, dropout_rate, input_data_adj, input_data_gene_feature, input_data_dense, gene_names_num,
                   input_dim=cell_names_num, output_dim=cell_label_num, logging=True)

# Initialize session
sess = tf.Session()
sess.run(tf.variables_initializer(list(input_data_adj[0].values())),
         feed_dict=feed_dict_adj[0])
sess.run(tf.variables_initializer(
    list(input_data_gene_feature.values())), feed_dict=feed_dict_gene_feature)
sess.run(tf.variables_initializer(
    list(input_data_dense.values())), feed_dict=feed_dict_dense)

# Init variables
sess.run(tf.global_variables_initializer())

val_out_dir = os.path.join(out_dir, "val_out")
if os.path.exists(val_out_dir) == False:
    os.makedirs(val_out_dir)

log_list = []
val_acc_list = []

train_embs = []
val_embs = []

best_val_model = None

t = time.time()

# Train model
for epoch in range(FLAGS.epochs):

    tot_train_loss = 0.
    pred_probas = []
    label = []
    while True:
        a = train_dataset.next()
        ind, label_p = a
        if ind is None:
            train_dataset.reset()
            break
        # Training step
        feed_dict = {index: ind, dropout_rate: FLAGS.dropout}
        _, b_cell_emb, b_loss, b_acc, b_probas = sess.run([model.opt_op, model.cell_emb, model.loss, model.accuracy, model.probas], feed_dict=feed_dict)
        pred_probas.append(b_probas)
        tot_train_loss += b_loss* len(ind)
        label.append(label_p)
    pred_probas = np.concatenate(pred_probas, axis=0)
    label = np.concatenate(label, axis=0)
    train_acc = accuracy(pred_probas, label)
    train_kappa = kappa_score(pred_probas, label)
    avg_train_loss = tot_train_loss / train_dataset.samples


    tot_val_loss = 0.
    pred_probas = []

    while True:
        a = val_dataset.next()
        ind, _ = a
        if ind is None:
            val_dataset.reset()
            break
        # Training step
        feed_dict = {index: ind}
        batch_val_loss, probas = sess.run([model.loss, model.probas], feed_dict=feed_dict)
        pred_probas.append(probas)
        tot_val_loss += batch_val_loss * len(ind)
    pred_probas = np.concatenate(pred_probas, axis=0)

    val_pred_cls = np.argmax(pred_probas, axis=1).reshape(-1,1)
    val_true_cls = np.argmax(val_dataset.label[sum(cell_set_size[0:3]):], axis=1).reshape(-1,1)
    val_out = np.concatenate([val_true_cls, val_pred_cls, pred_probas], axis=1)
    val_df = pd.DataFrame(val_out, columns=["true", "pred"] + cell_labels)
    val_path = os.path.join(val_out_dir, "val_{}.csv".format(epoch))
    val_df.to_csv(val_path, index=False, encoding="utf-8")

    val_acc = accuracy(pred_probas, val_dataset.label[sum(cell_set_size[0:3]):])
    val_kappa = kappa_score(pred_probas, val_dataset.label[sum(cell_set_size[0:3]):])
    avg_val_loss = tot_val_loss / val_dataset.samples
    val_acc_list.append(val_acc)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.3f}".format(avg_train_loss),
          "train_acc=", "{:.3f}".format(train_acc), "train_kappa=", "{:.3f}".format(train_kappa),
          "val_loss=", "{:.3f}".format(avg_val_loss),
          "val_acc=", "{:.3f}".format(val_acc),  "val_kappa=", "{:.3f}".format(val_kappa),
          "time=", "{:.3f}".format(time.time() - t))

    log_list.append([avg_train_loss, train_acc, train_kappa, avg_val_loss, val_acc, val_kappa])

    if epoch > FLAGS.early_stopping and val_acc_list[-1] > np.mean(val_acc_list[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

t1 = time.time()
print("Optimization Finished!")
print("total cost time {}".format(t1 - t))


log_df = pd.DataFrame(log_list, columns=["train_loss", "train_acc", "train_kappa", "val_loss", "val_acc", "val_kappa"])
log_path = os.path.join(out_dir, "log.csv")
log_df.to_csv(log_path, index=False, encoding="utf-8")


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
        test_dataset.reset()
        break
    feed_dict = {index: ind}
    test_loss, probas = sess.run([model.loss, model.probas], feed_dict=feed_dict)
    pred_probas.append(probas)
    tot_test_loss += test_loss * len(ind)
pred_probas = np.concatenate(pred_probas, axis=0)
test_acc = accuracy(pred_probas, test_dataset.label[:cell_set_size[0]])
test_kappa = kappa_score(pred_probas, test_dataset.label[:cell_set_size[0]])
avg_test_loss = tot_test_loss / test_dataset.samples

cost_test.append(avg_test_loss)

# Print results
print("test_acc=", "{:.5f}".format(test_acc), "test_kappa=", "{:.5f}".format(test_kappa), "time=", "{:.5f}".format(time.time() - t))
