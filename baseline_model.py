import dataholder
import numpy as np
import tensorflow as tf

from utils import *


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


class LSTM_Baseline_Model:
    def __init__(self, is_training=0):
        self.output_num = 3

        self.keep_prob = 0.75
        self.drop1 = 0.05
        self.drop2 = 0.1
        self.drop3 = 0.15

        self.h1 = 200
        self.h2 = 200

        if is_training != 0:
            self.keep_prob = 1.0
            self.drop1 = 0
            self.drop2 = 0
            self.drop3 = 0

        """
        변수목록

        Embedding Vector
        WSD Embedding Vector
        Input
        Label
        Label Helper

        """

        self.processor = dataholder.DataHolder(is_Training=is_training)
        if is_training != 0:
            self.keep_prob = 1.0

        # Embedding Lookup 해야함
        self.Sequence = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.Label = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
        self.Label_Helper = tf.placeholder(shape=[None, None, 1], dtype=tf.float32)
        self.POS_Sequence = tf.placeholder(shape=[None, None], dtype=tf.int32)

        ################### word2vec tensor###########
        self.word_embedding_kor = tf.get_variable(
            "word2vec_kor", initializer=tf.constant(self.processor.embedding_arr, dtype=tf.float32), trainable=False)

    def forward(self):
        pos_embedding = tf.get_variable(
            name="pos_embedding",
            shape=[71, 32],
            initializer=create_initializer(0.02))

        pos1 = tf.nn.embedding_lookup(pos_embedding, self.POS_Sequence)

        sequence = tf.nn.embedding_lookup(self.word_embedding_kor, self.Sequence)
        sequence = tf.concat([pos1, sequence], axis=2)

        # LSTM Cells
        cell_fw_L1 = tf.nn.rnn_cell.GRUCell(self.h1)
        cell_fw_L1 = tf.nn.rnn_cell.DropoutWrapper(cell_fw_L1, input_keep_prob=self.keep_prob,
                                                   output_keep_prob=self.keep_prob)
        cell_fw_L2 = tf.nn.rnn_cell.GRUCell(self.h2)
        cell_fw_L2 = tf.nn.rnn_cell.DropoutWrapper(cell_fw_L2, input_keep_prob=self.keep_prob,
                                                   output_keep_prob=self.keep_prob)

        cell_bw_L1 = tf.nn.rnn_cell.GRUCell(self.h1)
        cell_bw_L1 = tf.nn.rnn_cell.DropoutWrapper(cell_bw_L1, input_keep_prob=self.keep_prob,
                                                   output_keep_prob=self.keep_prob)
        cell_bw_L2 = tf.nn.rnn_cell.GRUCell(self.h2)
        cell_bw_L2 = tf.nn.rnn_cell.DropoutWrapper(cell_bw_L2, input_keep_prob=self.keep_prob,
                                                   output_keep_prob=self.keep_prob)

        with tf.variable_scope("rnn_model1"):
            (output_fw, output_bw), state_fw = tf.nn.bidirectional_dynamic_rnn(
                inputs=sequence, cell_fw=cell_fw_L1, cell_bw=cell_bw_L1,
                sequence_length=seq_length(sequence), dtype=tf.float32, time_major=False)

            output = tf.concat([output_fw, output_bw], axis=2)
        with tf.variable_scope("rnn_model2"):
            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                inputs=output, cell_fw=cell_fw_L2, cell_bw=cell_bw_L2,
                sequence_length=seq_length(output), dtype=tf.float32, time_major=False)

            output = tf.concat([output_fw, output_bw], axis=2)
        with tf.variable_scope("rnn_model2"):
            output_num = self.output_num

            hidden_states_ = Fully_Connected(output, output=300, name='hidden_layer', activation=tf.nn.tanh,
                                             reuse=False)

            hidden_states_ = tf.nn.dropout(hidden_states_, self.keep_prob + self.drop1)

            hidden_states_ = Highway_Network_Fullyconnceted(x=hidden_states_, dropout=0.2, name='hidden_layer1',
                                                            size=300, padding=False)
            hidden_states_ = tf.nn.dropout(hidden_states_, self.keep_prob + self.drop2)

            # hidden_states_ = Highway_Network_Fullyconnceted(x=hidden_states_, dropout=0.2, name='hidden_layer2',
            #                                                size=300, padding=False)
            hidden_states_ = tf.nn.dropout(hidden_states_, self.keep_prob + self.drop3)

            #hidden_states = Fully_Connected(hidden_states_, output=200, name='output_layer', activation=tf.nn.tanh,
            #                                reuse=False)

            #Task에 따라서 달라짐
            output_num = output_num
            prediction = Fully_Connected(hidden_states_, output=output_num, name='prediction_layer', activation=None,
                                         reuse=False)

            #masking = tf.tile(self.Label_Helper, multiples=[1, 1, output_num])
            prediction = tf.multiply(prediction, self.Label_Helper)

            return prediction

    def Training(self, epoch, is_continue, l2_norm=3e-7, gradient=True):
        save_path = 'C:\\Users\\USER\\Desktop\\reading_punc\\rp_model'

        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.95

        with tf.Session() as sess:
            prediction = self.forward()

            total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.Label, logits=prediction)

            regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)
            if l2_norm is not None:
                variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                total_loss += l2_loss

            total_loss = tf.reduce_mean(total_loss)

            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.001
            grad_clip = 5.0
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       200, 0.99, staircase=True)

            if gradient is True:
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-7)
                grads = opt.compute_gradients(total_loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(
                    gradients, grad_clip)
                optimizer = opt.apply_gradients(
                    zip(capped_grads, variables), global_step=global_step)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

            sess.run(tf.initialize_all_variables())

            if is_continue is True:
                print('restored!')
                saver = tf.train.Saver()
                saver.restore(sess, save_path)

            epo = 0

            while epo < epoch:
                sequence, label_symbol, label_mag, label_masking, pos_sequence = self.processor.next_batch()
                training_feed_dict = {self.Sequence: sequence, self.Label: label_mag,
                                      self.Label_Helper: label_masking, self.POS_Sequence: pos_sequence}

                epo += 1

                sess.run(optimizer, feed_dict=training_feed_dict)

                if epo % 100 == 0:
                    _, loss_value = sess.run([optimizer, total_loss], feed_dict=training_feed_dict)
                    print(epo, ',', ':', loss_value)

                if epo % 500 == 0:
                    saver = tf.train.Saver()
                    saver.save(sess, save_path)
                    print('saved!')

            saver = tf.train.Saver()
            saver.save(sess, save_path)

        return 0

    def eval(self, log_write=False):
        label_words = []
        label_words.append('#')
        label_words.append('$')
        label_words.append('//H')
        label_words.append('///H')
        label_words.append('//M')
        label_words.append('///M')
        label_words.append('//L')
        label_words.append('///L')

        cnt_arr = np.zeros(shape=[8], dtype=np.int32)
        cor_arr = np.zeros(shape=[8], dtype=np.int32)

        if log_write is True:
            log_file = open('log_result', 'w', encoding='utf-8')

        save_path = 'C:\\Users\\USER\\Desktop\\reading_punc\\rp_model'

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        with tf.Session(config=config) as sess:
            prediction = self.forward()
            prediction_idx = tf.argmax(prediction, axis=2)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, save_path)

            epo = 0

            cor = 0
            cnt = 0

            for b in range(1):
                sequence, label_symbol, label_mag, label_masking, pos_sequence = self.processor.test_batch()
                """
                print(label[0])
                for i in range(label_helper.shape[1]):
                    if i == label[0]:
                        print('1,', label_helper[0, i])
                    else:
                        print('0,', label_helper[0, i])
                input()
                """
                # print(label.shape)
                # print(sequence1.shape)
                # print(sequence2.shape)
                # print(label_helper.shape)
                training_feed_dict = {self.Sequence: sequence, self.Label_Helper: label_masking,
                                      self.POS_Sequence: pos_sequence}

                epo += 1

                result = sess.run(prediction_idx, feed_dict=training_feed_dict)
                result = np.array(result, dtype=np.int32)

                for i in range(result.shape[0]):
                    for j in range(result.shape[1]):
                        if label_masking[i, j, 0] == 1:
                            #if label_symbol[i, j] != 0:
                            if result[i, j] == label_mag[i, j]:
                                cor += 1
                                cor_arr[label_mag[i, j]] += 1
                            cnt += 1
                            cnt_arr[label_mag[i, j]] += 1

        print(result.shape)
        print(label_symbol.shape)
        print(cor, '/', cnt)
        for i in range(self.output_num):
            print(label_words[i], ':', cor_arr[i], '/', cnt_arr[i], ' acc:', (cor_arr[i] / cnt_arr[i]))

        return 0

    def propagate(self):
        label_words = []
        label_words.append('#')
        label_words.append('//')
        label_words.append('///')

        cnt_arr = np.zeros(shape=[8], dtype=np.int32)
        cor_arr = np.zeros(shape=[8], dtype=np.int32)

        save_path = 'C:\\Users\\USER\\Desktop\\reading_punc\\rp_model'

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        with tf.Session(config=config) as sess:
            prediction = self.forward()
            prediction_idx = tf.argmax(prediction, axis=2)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()
            saver.restore(sess, save_path)

            epo = 0

            cor = 0
            cnt = 0

            while True:
                print('테스트 할 문장을 입력하세요')
                s = input()

                sequence, pos_sequence, label_masking, morphs = self.processor.sentence_batch(s)

                training_feed_dict = {self.Sequence: sequence, self.Label_Helper: label_masking,
                                      self.POS_Sequence: pos_sequence}

                epo += 1

                result = sess.run(prediction_idx, feed_dict=training_feed_dict)
                result = np.array(result, dtype=np.int32)

                for i in range(len(morphs)):
                    print(morphs[i], ':', label_words[result[0, i]])
        return 0
