import tensorflow as tf
import numpy as np
from mha import MultiHeadedAttention

#tf.compat.v1.disable_eager_execution() # for v1.compat.placeholder not in v2

class HierarchicalLSTM:
    def __init__(self, pc_vocab_size, page_vocab_size, page_out_vocab_size, batch_size, num_steps, learning_rate, pc_embed_size, page_embed_size, offset_embed_size, offset_embed_size_internal, lstm_size=128, num_layers=2):
        """
        Create the model inputs
        """
        print('initalizing...')
        self.offset_size = 64
        self.pc_embed_size = pc_embed_size
        self.page_embed_size = page_embed_size
        self.offset_embed_size = offset_embed_size
        self.offset_embed_size_internal = offset_embed_size_internal
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.pc_vocab_size = pc_vocab_size
        self.page_vocab_size = page_vocab_size
        self.page_out_vocab_size = page_out_vocab_size
        self.batch_size = batch_size

        print('PC embed size: {}'.format(self.pc_embed_size))
        print('Page embed size: {}'.format(self.page_embed_size))
        print('Offset embed size: {}'.format(self.offset_embed_size))
        print('Coarse lstm size: {}'.format(self.lstm_size))
        print('Fine lstm size: {}'.format(self.lstm_size))

        # pc_in, page_in and offset_in: value are integers
        self.pc_in = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.num_steps], name='pc_in')

        self.pl_page_in = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.num_steps], name='pl_page_in')
        self.pl_offset_in = tf.compat.v1.placeholder(tf.int32, [self.batch_size, self.num_steps], name='pl_offset_in')

        # page_out (labels) and offset_out (labels): values are one_hot
        self.page_out_init = tf.compat.v1.placeholder(tf.int32, [self.batch_size], name='page_out_init')
        self.page_out = tf.one_hot(self.page_out_init, self.page_out_vocab_size, name='page_out') # tf.one_hot(indices, depth)
        self.offset_out_init = tf.compat.v1.placeholder(tf.int32, [self.batch_size], name='offset_out_init')
        self.offset_out = tf.one_hot(self.offset_out_init, self.offset_size, name='offset_out')

        # others
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, shape=(), name="keep_ratio")
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=(), name="learning_rate")

        self.build_embedding_layer()

        self.build_page_offset_attention()

        self.lstm_inputs = tf.concat([self.pc_embed, self.pl_page_embed, self.pl_offset_embed], 2)

        assert np.shape(self.lstm_inputs) == [self.batch_size, self.num_steps, self.pc_embed_size+self.page_embed_size*2]

        print('Input feature size: ', np.shape(self.lstm_inputs))

        self.build_lstm_layers()
        self.build_cost_fn_and_opt()
        self.build_accuracy()

    def build_page_offset_attention(self):
        print('Conditional attention embedding')
        mha1 = MultiHeadedAttention(d_model=self.page_embed_size, num_heads=1)

        # reshape self.pl_offset_embed to 
        # (self.batch_size, self.num_steps, self.offset_embed_size//self.page_embed_size, self.page_embed_size)
        self.pl_offset_embed = tf.reshape(self.pl_offset_embed, shape=(self.batch_size, self.num_steps, self.offset_embed_size // self.page_embed_size, self.page_embed_size))
        pl_page_embed = tf.reshape(self.pl_page_embed, shape=(self.batch_size, self.num_steps, 1, self.page_embed_size))

        self.pl_offset_embed, self.attns = mha1.call(query=pl_page_embed, key=self.pl_offset_embed, value=self.pl_offset_embed)


    def build_embedding_layer(self):
        """
        Create the embedding layer
        """
        print('building embedding layer...')
        self.pc_embedding = tf.Variable(tf.random.uniform((self.pc_vocab_size, self.pc_embed_size), -1.0, 1.0), trainable=True) #Outputs random values from a uniform 
        #distribution in range -1.0 to 1.0 with shape as the first parameter 
        self.page_embedding = tf.Variable(tf.random.uniform((self.page_vocab_size, self.page_embed_size), -1.0, 1.0), trainable=True)
        self.offset_embedding = tf.Variable(tf.random.uniform((self.offset_size, self.offset_embed_size), -1.0, 1.0), trainable=True)

        self.pc_embed = tf.nn.embedding_lookup(params=self.pc_embedding, ids=self.pc_in) #parallel lookups in params (1st parameter) by IDs (2nd parameter)
        self.pl_page_embed = tf.nn.embedding_lookup(params=self.page_embedding, ids=self.pl_page_in)
        self.pl_offset_embed = tf.nn.embedding_lookup(params=self.offset_embedding, ids=self.pl_offset_in)


    def build_lstm_layers(self):
        """
        Create the LSTM layers
        """
        # page number prediction
        print('building lstm layer...')
        coarse_cells = [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.lstm_size) for i in range(self.num_layers)]
        coarse_cells = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in coarse_cells]
        coarse_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(coarse_cells, state_is_tuple=True)
        coarse_outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell=coarse_cells, inputs=self.lstm_inputs, scope="coarse", dtype=tf.float32) #output uses a list of lstm inputs
        assert np.shape(coarse_outputs) == [self.batch_size, self.num_steps, self.lstm_size]

        self.coarse_lstm_outputs = coarse_outputs[:, -1]

        # page offset prediction
        print('building lstm layer...')
        fine_cells = [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.lstm_size) for i in range(self.num_layers)]
        fine_cells = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob) for cell in fine_cells]
        fine_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(fine_cells, state_is_tuple=True)
        fine_outputs, _ = tf.compat.v1.nn.dynamic_rnn(cell=fine_cells, inputs=self.lstm_inputs, scope="fine", dtype=tf.float32)
        assert np.shape(fine_outputs) == [self.batch_size, self.num_steps, self.lstm_size]

        self.fine_lstm_outputs = fine_outputs[:, -1]


    def build_cost_fn_and_opt(self):
        """
        Create the Loss function and Optimizer
        """
        print('building loss...')
        self.page_logits = tf.compat.v1.layers.dense(self.coarse_lstm_outputs, self.page_out_vocab_size, activation=None)
        self.offset_logits = tf.compat.v1.layers.dense(self.fine_lstm_outputs, self.offset_size, activation=None)
        assert np.shape(self.page_logits) == [self.batch_size, self.page_out_vocab_size]
        assert np.shape(self.offset_logits) == [self.batch_size, self.offset_size]

        page_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(self.page_out), logits=self.page_logits)
        offset_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(self.offset_out), logits=self.offset_logits)

        self.page_loss = tf.reduce_mean(input_tensor=page_loss)
        self.offset_loss = tf.reduce_mean(input_tensor=offset_loss)

        self.loss = self.page_loss + self.offset_loss
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.optimizer1 = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def build_accuracy(self):
        """
        Create accuracy
        """
        print('building accuracy...')
        self.page_preds = tf.argmax(input=self.page_logits, axis=1, name='page_predictions')
        self.offset_preds = tf.argmax(input=self.offset_logits, axis=1, name='offset_predictions')

        page_equal = tf.equal(self.page_preds, tf.argmax(input=self.page_out, axis=1))
        offset_equal = tf.equal(self.offset_preds, tf.argmax(input=self.offset_out, axis=1))

        self.page_accuracy = tf.reduce_mean(input_tensor=tf.cast(page_equal, "float"), name='page_accuracy')
        self.offset_accuracy = tf.reduce_mean(input_tensor=tf.cast(offset_equal, "float"), name='offset_accuracy')
        self.overall_accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.logical_and(page_equal, offset_equal), "float"), name='overall_accuracy')
