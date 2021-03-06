#encoding=utf-8
import os
import pandas as pd
from gcn_chi.layers import *
from gcn_chi.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, save_path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        # save_path = os.path.join(save_dir, "%s.ckpt" % self.name)
        saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def load(self, save_path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        # save_path = os.path.join(save_dir, "%s.ckpt" % self.name)
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class CellGeneGCN(Model):
  def __init__(self, index, dropout_rate, input_data_adj, input_data_gene_feature, input_data_dense, gene_names_num, input_dim,
               output_dim, **kwargs):
    super(CellGeneGCN, self).__init__(**kwargs)
    self.input_data_adj = input_data_adj
    self.input_data_gene_feature = input_data_gene_feature
    self.input_data_dense = input_data_dense

    self.gene_features = tf.SparseTensor(tf.to_int64(tf.stack([input_data_gene_feature["coo_row_ind"],
                                                               input_data_gene_feature["coo_col_ind"]], axis=1)),
                                         input_data_gene_feature["coo_val"], [gene_names_num, input_dim])
    self.adj = [tf.SparseTensor(tf.to_int64(tf.stack([input_data_adj[0]["coo_row_ind"],
                                                                  input_data_adj[0]["coo_col_ind"]], axis=1)),
                                            input_data_adj[0]["coo_val"], [gene_names_num, gene_names_num])]
    self.cell_gene_weight = input_data_dense['cell_gene_weight']
    self.cell_gene_weight = tf.gather(self.cell_gene_weight, index)
    self.cell_labels = tf.gather(self.input_data_dense['cell_labels'], index)
    self.labels_mask = tf.gather(self.input_data_dense['labels_mask'], index)
    self.input_dim = input_dim
    self.gene_names_num = gene_names_num
    # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
    self.output_dim = output_dim
    self.step = tf.Variable(0, trainable=False)
    self.rate = tf.train.exponential_decay(FLAGS.learning_rate, self.step, 1, 0.9999)
    self.dropout_rate = dropout_rate

    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.rate)

    self.build()

  def _loss(self):
    # Weight decay loss
    for var in self.gcn_layer.vars.values():
      self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    # Cross entropy error
    self.loss += masked_softmax_cross_entropy(self.outputs, self.cell_labels,
                                              self.labels_mask)


  def _accuracy(self):
    self.accuracy = masked_accuracy(self.probas, self.cell_labels,
                                    self.labels_mask)

  def _build(self):

    self.gcn_layer = GraphConvolution(input_dim=self.input_dim,
                                        output_dim=FLAGS.hidden1,
                                        input_data_adj=self.adj,
                                        input_data_dense=self.input_data_dense,
                                        act=tf.nn.relu,
                                        dropout=self.dropout_rate,
                                        sparse_inputs=True,
                                        logging=self.logging)

    # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
    #                                     output_dim=self.output_dim,
    #                                     placeholders=self.placeholders,
    #                                     act=lambda x: x,
    #                                     dropout=True,
    #                                     logging=self.logging))

  def build(self):

    """ Wrapper for _build() """
    with tf.variable_scope(self.name):
      self._build()
      gene_emb = self.gcn_layer(self.gene_features)
      cell_emb = tf.matmul( self.cell_gene_weight, gene_emb)
      self.cell_emb = cell_emb
      cell_activation = tf.nn.relu(cell_emb)
      cell_emb1 = tf.layers.dense(cell_activation, 32, activation=tf.nn.relu)
      self.outputs = tf.layers.Dense(self.output_dim)(cell_emb1)
      self.probas = tf.nn.softmax(self.outputs)

    # self.activations.append(self.inputs)
    # for layer in self.layers:
    #   hidden = layer(self.activations[-1])
    #   self.activations.append(hidden)
    # self.outputs = self.activations[-1]

    # Store model variables for easy access
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    self.vars = {var.name: var for var in variables}

    # Build metrics
    self._loss()
    self._accuracy()

    self.opt_op = self.optimizer.minimize(self.loss, global_step=self.step)

  def predict(self):
    return tf.nn.softmax(self.outputs)


def ValModel(sess, model_cls, val_dataset, index_placeholder, cell_labels, cell_set_size):
    tot_val_loss = 0.
    pred_probas = []
    while True:
      a = val_dataset.next()
      ind, _ = a
      if ind is None:
        val_dataset.reset()
        break
      # Training step
      feed_dict = {index_placeholder: ind}
      batch_val_loss, probas = sess.run([model_cls.loss, model_cls.probas], feed_dict=feed_dict)
      pred_probas.append(probas)
      tot_val_loss += batch_val_loss * len(ind)
    pred_probas = np.concatenate(pred_probas, axis=0)

    val_pred_cls = np.argmax(pred_probas, axis=1).reshape(-1, 1)
    val_true_cls = np.argmax(val_dataset.label[sum(cell_set_size[0:3]):], axis=1).reshape(-1, 1)
    val_out = np.concatenate([val_true_cls, val_pred_cls, pred_probas], axis=1)
    val_df = pd.DataFrame(val_out, columns=["true", "pred"] + cell_labels)
    return val_df

    # val_path = os.path.join(val_out_dir, "val_{}.csv".format(epoch))
    # val_df.to_csv(val_path, index=False, encoding="utf-8")

