# coding=utf-8
__author__ = 'yhd'

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.platform import gfile
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import platform

if platform.system() == 'Windows':
    from yhd.reader import *
    from yhd.iterator import *
else:
    from reader import *
    from iterator import *

import random
import copy
import re

BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_SIZE = 300
VOCAB_SIZE = 19495

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")

import codecs

class DataLoader(object):

    def __init__(self, is_toy=False):
        if is_toy:
            self.source_train = 'data_root/train.txt'
            self.source_test = 'data_root/test.txt'
            self.batch_size = 3
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = 'data_root/val.txt'
            self.test_batch_size = 3
            self.val_batch_size = 3
            self.source_train_act = 'data_root/act_train.txt'
            self.source_val_act = 'data_root/act_val.txt'
            self.source_test_act = 'data_root/act_test.txt'
        else:
            self.source_train = 'data_root/dialogues_train.txt'
            self.source_test = 'data_root/dialogues_test.txt'
            self.batch_size = BATCH_SIZE
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = 'data_root/dialogues_validation.txt'
            self.test_batch_size = BATCH_SIZE
            self.val_batch_size = BATCH_SIZE
            self.source_train_act = 'data_root/dialogues_emotion_train.txt'
            self.source_val_act = 'data_root/dialogues_emotion_validation.txt'
            self.source_test_act = 'data_root/dialogues_emotion_test.txt'


        if platform.system() == 'Windows':
            with open(self.source_train, 'r', encoding='utf-8') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.source_train_act, 'r', encoding='utf-8') as stf:
                self.train_act_raw_text = stf.readlines()

            with open(self.source_validation, 'r', encoding='utf-8') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.source_val_act, 'r', encoding='utf-8') as svf:
                self.validation_act_raw_text = svf.readlines()

            with open(self.source_test, 'r', encoding='utf-8') as stef:
                self.test_raw_text = stef.readlines()

            with open(self.source_test_act, 'r', encoding='utf-8') as stef:
                self.test_act_raw_text = stef.readlines()
        else:
            with open(self.source_train, 'r') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.source_train_act, 'r') as stf:
                self.train_act_raw_text = stf.readlines()

            with open(self.source_validation, 'r') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.source_val_act, 'r') as svf:
                self.validation_act_raw_text = svf.readlines()

            with open(self.source_test, 'r') as stef:
                self.test_raw_text = stef.readlines()

            with open(self.source_test_act, 'r') as stef:
                self.test_act_raw_text = stef.readlines()


        self.batch_num = len(self.train_raw_text) // self.batch_size
        self.val_batch_num = len(self.validation_raw_text) // self.val_batch_size
        self.test_batch_num = len(self.test_raw_text) // self.test_batch_size

        self.train_pointer = 0
        self.val_pointer = 0
        self.test_pointer = 0


        self.initialize_vocabulary()

    def initialize_vocabulary(self, vocabulary_path='data_root/vocab50000.in'):
      """Initialize vocabulary from file.

      We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
      will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
      also return the reversed-vocabulary ["dog", "cat"].

      Args:
        vocabulary_path: path to the file containing the vocabulary.

      Returns:
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).

      Raises:
        ValueError: if the provided vocabulary_path does not exist.
      """
      if gfile.Exists(vocabulary_path):
        rev_vocab = []

        with gfile.GFile(vocabulary_path, mode="r") as f:
          rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

        self.vocab_id = vocab
        self.id_vocab = {v: k for k, v in vocab.items()}
        self.rev_vocab = rev_vocab

    def basic_tokenizer(self, sentence):
      """Very basic tokenizer: split the sentence into a list of tokens."""
      words = []
      for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
      return [w.lower() for w in words if w]

    def sentence_to_token_ids(self, sentence, tokenizer=None, normalize_digits=True):
      """Convert a string to list of integers representing token-ids.

      For example, a sentence "I have a dog" may become tokenized into
      ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
      "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

      Args:
        sentence: a string, the sentence to convert to token-ids.
        vocabulary: a dictionary mapping tokens to integers.
        tokenizer: a function to use to tokenize each sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.

      Returns:
        a list of integers, the token-ids for the sentence.
      """
      if tokenizer:
        words = tokenizer(sentence)
      else:
        words = self.basic_tokenizer(sentence)
      if not normalize_digits:
        return [self.vocab_id.get(w, UNK_ID) for w in words]
      # Normalize digits by 0 before looking words up in the vocabulary.
      sentence_ids = [self.vocab_id.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]
      return sentence_ids

    def load_embedding(self, embedding_file='glove/glove.840B.300d.txt'):
        embedding_index = {}
        f = open(embedding_file)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
        embedding_index['_PAD'] = np.zeros(300, dtype=np.float32)
        embedding_index['_GO'] = np.zeros(300, dtype=np.float32)
        embedding_index['_EOS'] = np.zeros(300, dtype=np.float32)
        lookup_table = []
        num = 0
        sorted_keys = [k for k in sorted(self.id_vocab.keys())]
        for w_id in sorted_keys:
            if self.id_vocab[w_id] in embedding_index:
                num += 1
                lookup_table.append(embedding_index[self.id_vocab[w_id]])
            else:
                lookup_table.append(embedding_index['unk'])

        f.close()
        print("Total {}/{} words vector.".format(num, len(self.id_vocab)))
        self.embedding_matrix = lookup_table

    def batch_contents_into_items(self, dialogues, dia_acts):
        sen_list = []
        act_list = []
        for idx, ele in enumerate(dia_acts):
            sen_acts = ele.split()
            sentences = dialogues[idx].split('__eou__')[:-1]
            for jdx, item in enumerate(sen_acts):
                sen_list.append(self.sentence_to_token_ids(sentences[jdx]))
                act_list.append(int(item))
        return sen_list, act_list

    def get_batch_test(self):
        if self.test_pointer < self.test_batch_num:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
            act_raw_data = self.test_act_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
        else:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size: ]
            act_raw_data = self.test_act_raw_text[self.test_pointer * self.test_batch_size: ]

        self.test_pointer += 1

        sen_batch, class_batch = self.batch_contents_into_items(raw_data, act_raw_data)
        sen_batch, class_batch = np.asarray(sen_batch), np.asarray(class_batch)
        if sen_batch.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None])

        sen_length = [len(item) for item in sen_batch]

        return np.asarray(self.pad_sentence(sen_batch, np.amax(sen_length))), np.asarray(sen_length), \
                np.asarray(class_batch)

    def get_batch_data(self):
        if self.train_pointer < self.batch_num:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
            act_raw_data = self.train_act_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
        else:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size: ]
            act_raw_data = self.train_act_raw_text[self.train_pointer * self.batch_size: ]

        self.train_pointer += 1

        sen_batch, class_batch = self.batch_contents_into_items(raw_data, act_raw_data)
        sen_batch, class_batch = np.asarray(sen_batch), np.asarray(class_batch)
        if sen_batch.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None])

        sen_length = [len(item) for item in sen_batch]

        return np.asarray(self.pad_sentence(sen_batch, np.amax(sen_length))), np.asarray(sen_length), \
                np.asarray(class_batch)

    def get_validation(self):
        if self.val_pointer < self.val_batch_num:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
            act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
        else:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size: ]
            act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size: ]

        self.val_pointer += 1

        sen_batch, class_batch = self.batch_contents_into_items(raw_data, act_raw_data)
        sen_batch, class_batch = np.asarray(sen_batch), np.asarray(class_batch)
        if sen_batch.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None])

        sen_length = [len(item) for item in sen_batch]

        return np.asarray(self.pad_sentence(sen_batch, np.amax(sen_length))), np.asarray(sen_length), \
                np.asarray(class_batch)

    def pad_sentence(self, sentences, max_length):
        pad_sentences = []
        for sentence in sentences:
            if len(sentence) > max_length:
                sentence = sentence[0: max_length]
            else:
                for _ in range(len(sentence), max_length):
                    sentence.append(PAD_ID)
            pad_sentences.append(sentence)
        return pad_sentences


    def reset_pointer(self):
        self.train_pointer = 0
        self.val_pointer = 0


class Seq2seq(object):

    def __init__(self, num_layers=1):
        self.embedding_size = EMBEDDING_SIZE
        self.vocab_size = VOCAB_SIZE
        self.act_class_num = 7

        self.create_model()

    def create_model(self):
        self.sen_input = tf.placeholder(tf.int32, [None, None], name='sen_input')
        self.sen_input_lengths = tf.placeholder(tf.int32, [None], name='sen_input_lengths')
        self.class_label = tf.placeholder(tf.int64, [None], name='class_label')

        self.dropout_kp = tf.placeholder(tf.float32, name='dropout_kp', shape=())

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.constant(0., shape=[self.vocab_size, self.embedding_size]), name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size],
                                                        name='embedding_placeholder')
            embeding_init = W.assign(self.embedding_placeholder)
            sen_embedded_inputs = tf.nn.embedding_lookup(embeding_init, self.sen_input)

        attention = tf.sigmoid(sen_embedded_inputs)
        self.weights_for_reward = tf.reduce_mean(attention, axis=-1, name='weights_for_reward')
        dense_inputs = tf.reduce_mean(attention * sen_embedded_inputs, axis=1)

        layer1_units = 100
        layer1 = Dense(layer1_units)
        layer2 = Dense(self.act_class_num)

        h1 = layer1(dense_inputs)
        h2 = layer2(h1)

        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(h2), axis=1), self.class_label)
        self.accuracy = tf.cast(correct_prediction, "float", name='accuracy')

        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h2, labels=self.class_label)
        )

        with tf.device('/cpu:0'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)

    def train(self, sess, sen_input, sen_input_lengths, class_label,
                      embedding_placeholder, dropout_kp):
        _, loss = sess.run([self.train_op, self.cost],
                                   feed_dict={self.sen_input: sen_input,
                                              self.sen_input_lengths: sen_input_lengths,
                                              self.class_label: class_label,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp})
        return loss

    def validation(self, sess, sen_input, sen_input_lengths, class_label,
                      embedding_placeholder, dropout_kp):
        loss, acc = sess.run([self.cost, self.accuracy],
                                   feed_dict={self.sen_input: sen_input,
                                              self.sen_input_lengths: sen_input_lengths,
                                              self.class_label: class_label,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp})
        return loss, acc

    def visualization(self, sess, merged, sen_input, sen_input_lengths, class_label,
                      embedding_placeholder, dropout_kp):
        loss = sess.run(merged, feed_dict={self.sen_input: sen_input,
                                              self.sen_input_lengths: sen_input_lengths,
                                              self.class_label: class_label,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp})
        return loss

    def get_train_logit(self, sess, sen_input, sen_input_lengths, class_label,
                      embedding_placeholder, dropout_kp):
        logits = sess.run(self.cost,
                                   feed_dict={self.sen_input: sen_input,
                                              self.sen_input_lengths: sen_input_lengths,
                                              self.class_label: class_label,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp})
        return logits




import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug


MAX_TO_KEEP = 50

EPOCH_SIZE = 50

prefix = 'attention_dan_64_emotion'

def main_train(is_toy=False):
    data_loader = DataLoader(is_toy)

    model = Seq2seq()

    log_file = 'log/log_' + prefix + '.txt'
    log = codecs.open(log_file, 'w')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    print('train')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    else:
        data_loader.load_embedding()
    print('load the embedding matrix')

    checkpoint_storage = 'models_' + prefix + '/checkpoint'
    checkpoint_dir = 'models_' + prefix + '/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    with tf.Session(config=config) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=MAX_TO_KEEP)
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('summary_' + prefix + '/', sess.graph)
        sess.run(tf.global_variables_initializer())
        # if os.path.exists(checkpoint_storage):
        #     checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        #     loader = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        #     loader.restore(sess, checkpoint_file)
        #     print('Model has been restored')

        loss_list = []
        # train
        for epoch in range(EPOCH_SIZE):
            losses = 0
            step = 0
            val_losses = 0
            val_step = 0
            accs = 0

            for _ in range(data_loader.batch_num + 1):
                pad_sen_batch, sen_length, act_label = data_loader.get_batch_data()

                if pad_sen_batch.all() == None:
                    continue
                step += 1
                loss_mean = model.train(sess, pad_sen_batch, sen_length, act_label,
                                        data_loader.embedding_matrix, 0.8)
                losses += loss_mean

                # if step % 2 == 0:
                #     result = model.visualization(sess, merged, pad_q_batch, q_length, pad_kb_batch, kb_length, go_pad_y_batch,
                #                         y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch)
                #     writer.add_summary(result, step)

            loss_list.append(losses / step)

            for _ in range(data_loader.val_batch_num + 1):
                pad_sen_batch, sen_length, act_label = data_loader.get_validation()
                if pad_sen_batch.all() == None:
                    continue
                val_loss_mean, acc = model.validation(sess, pad_sen_batch, sen_length, act_label,
                                        data_loader.embedding_matrix, 1)
                val_step += 1
                val_losses += val_loss_mean
                accs += np.mean(acc)

            print('step', step)
            print('val_step', val_step)

            print("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g} Valid Acc {:g}".format(epoch + 1,
                                        EPOCH_SIZE, losses / step, val_losses / val_step, accs / val_step))
            log.write("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g} Valid Acc {:g}\n".format(epoch + 1,
                                        EPOCH_SIZE, losses / step, val_losses / val_step, accs / val_step))

            saver.save(sess, checkpoint_prefix, global_step=epoch + 1)
            print('Model Trained and Saved in epoch ', epoch + 1)

            data_loader.reset_pointer()

        # plt.plot(loss_list)
        # plt.show()

        log.close()


if platform.system() == 'Windows':
    from yhd.bleu import *
    from yhd.perplexity import *
else:
    from bleu import *
    from perplexity import *

def main_test(is_toy=False):
    data_loader = DataLoader(is_toy)

    res_file = 'results/' + prefix + '_results.txt'
    res = codecs.open(res_file, 'w')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    test_graph = tf.Graph()

    with test_graph.as_default():
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # test
            print('test')
            if is_toy:
                data_loader.load_embedding('glove_false/glove.840B.300d.txt')
            else:
                data_loader.load_embedding()
            print('load the embedding matrix')

            checkpoint_file = 'models_' + prefix + '/model-1'

            loader = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            loader.restore(sess, checkpoint_file)
            print('Model has been restored')

            sen_input = test_graph.get_tensor_by_name('sen_input:0')
            sen_input_lengths = test_graph.get_tensor_by_name('sen_input_lengths:0')
            class_label = test_graph.get_tensor_by_name('class_label:0')
            dropout_kp = test_graph.get_tensor_by_name('dropout_kp:0')
            accuracy = test_graph.get_tensor_by_name('accuracy:0')
            embedding_placeholder = test_graph.get_tensor_by_name("embedding/embedding_placeholder:0")

            all_acc = 0
            test_step = 0

            for _ in range(data_loader.test_batch_num + 1):
                pad_sen_batch, sen_length, act_label = data_loader.get_batch_test()
                if pad_sen_batch.all() == None:
                    continue
                acc = sess.run(accuracy,
                                       feed_dict={sen_input: pad_sen_batch,
                                                  sen_input_lengths: sen_length,
                                                  class_label: act_label,
                                                  embedding_placeholder: data_loader.embedding_matrix,
                                                  dropout_kp: 1.0})
                all_acc += acc
                test_step += 1
                print('acc is ', acc)

            print('final acc is ', all_acc / test_step)

            res.close()


if __name__ == '__main__':
    # main_train(True)

    # main_test(True)

    main_train()

    # main_test()
