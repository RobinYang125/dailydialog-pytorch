# coding=utf-8
__author__ = 'yhd'

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.platform import gfile
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

BATCH_SIZE = 8
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
            self.source_train_emotion = 'data_root/emotion_train.txt'
            self.source_val_emotion = 'data_root/emotion_validation.txt'
            self.source_test_emotion = 'data_root/emotion_test.txt'
        else:
            self.source_train = 'data_root/data_train.txt'
            self.source_test = 'data_root/dialogues_test.txt'
            self.batch_size = BATCH_SIZE
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = 'data_root/data_val.txt'
            self.test_batch_size = BATCH_SIZE
            self.val_batch_size = BATCH_SIZE
            self.source_train_act = 'data_root/dialogues_act_train.txt'
            self.source_val_act = 'data_root/dialogues_act_validation.txt'
            self.source_test_act = 'data_root/dialogues_act_test.txt'
            self.source_train_emotion = 'data_root/dialogues_emotion_train.txt'
            self.source_val_emotion = 'data_root/dialogues_emotion_validation.txt'
            self.source_test_emotion = 'data_root/dialogues_emotion_test.txt'

        if platform.system() == 'Windows':
            with open(self.source_train, 'r', encoding='utf-8') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.source_train_act, 'r', encoding='utf-8') as stf:
                self.train_act_raw_text = stf.readlines()

            with open(self.source_train_emotion, 'r', encoding='utf-8') as stf:
                self.train_emotion_raw_text = stf.readlines()

            with open(self.source_validation, 'r', encoding='utf-8') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.source_val_act, 'r', encoding='utf-8') as svf:
                self.validation_act_raw_text = svf.readlines()

            with open(self.source_val_emotion, 'r', encoding='utf-8') as svf:
                self.validation_emotion_raw_text = svf.readlines()

            with open(self.source_test, 'r', encoding='utf-8') as stef:
                self.test_raw_text = stef.readlines()

            with open(self.source_test_act, 'r', encoding='utf-8') as stef:
                self.test_act_raw_text = stef.readlines()

            with open(self.source_test_emotion, 'r', encoding='utf-8') as stef:
                self.test_emotion_raw_text = stef.readlines()
        else:
            with open(self.source_train, 'r') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.source_train_act, 'r') as stf:
                self.train_act_raw_text = stf.readlines()

            with open(self.source_train_emotion, 'r') as stf:
                self.train_emotion_raw_text = stf.readlines()

            with open(self.source_validation, 'r') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.source_val_act, 'r') as svf:
                self.validation_act_raw_text = svf.readlines()

            with open(self.source_val_emotion, 'r') as svf:
                self.validation_emotion_raw_text = svf.readlines()

            with open(self.source_test, 'r') as stef:
                self.test_raw_text = stef.readlines()

            with open(self.source_test_act, 'r') as stef:
                self.test_act_raw_text = stef.readlines()

            with open(self.source_test_emotion, 'r') as stef:
                self.test_emotion_raw_text = stef.readlines()


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
                act_list.append(int(item) - 1)
        return sen_list, act_list

    def get_batch_act_data(self):
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

    def dialogues_into_qas(self, dialogues):
        qa_pairs = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')
            for i in range(len(sentences) - 2):
                qa = [sentences[i], sentences[i + 1]]
                qa_pairs.append([self.sentence_to_token_ids(sentences[i]),
                                      self.sentence_to_token_ids(sentences[i + 1])])
        return qa_pairs

    def dialogues_acts_emotions_into_qas(self, dialogues, dia_emotions, dia_acts):
        qa_pairs = []
        act_list = []
        emotion_list = []
        for idx, dialogue in enumerate(dialogues):
            sentences = dialogue.split('__eou__')[:-1]
            for i in range(len(sentences) - 1):
                qa = [sentences[i], sentences[i + 1]]
                qa_pairs.append([self.sentence_to_token_ids(sentences[i]),
                                      self.sentence_to_token_ids(sentences[i + 1])])

            emotions = dia_emotions[idx].split()
            for i in range(len(emotions) - 1):
                emotion_list.append([int(emotions[i]), int(emotions[i + 1])])

            acts = dia_acts[idx].split()
            for i in range(len(acts) - 1):
                act_list.append([int(acts[i]) - 1, int(acts[i + 1]) - 1])
        return qa_pairs, act_list, emotion_list

    def dialogues_into_qas_without_id(self, dialogues):
        qa_pairs = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')
            for i in range(len(sentences) - 2):
                qa = [sentences[i], sentences[i + 1]]
                qa_pairs.append(qa)
        return qa_pairs

    def get_batch_test(self):
        if self.test_pointer < self.test_batch_num:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
        else:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size: ]

        self.test_pointer += 1

        self.test_qa_pairs = np.asarray(self.dialogues_into_qas(raw_data))
        if self.test_qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None])

        self.test_raw_data = np.asarray(self.dialogues_into_qas_without_id(raw_data))
        self.test_x_batch = self.test_raw_data[:, 0]
        self.test_y_batch = self.test_raw_data[:, -1]

        x_batch = self.test_qa_pairs[:, 0]
        y_batch = self.test_qa_pairs[:, -1]

        x_length = [len(item) for item in x_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), np.asarray(x_length), \
                np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length)

    def get_batch_data(self):
        if self.train_pointer < self.batch_num:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
            act_raw_data = self.train_act_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
            emotion_raw_data = self.train_emotion_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
        else:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size: ]
            act_raw_data = self.train_act_raw_text[self.train_pointer * self.batch_size: ]
            emotion_raw_data = self.train_emotion_raw_text[self.train_pointer * self.batch_size: ]

        self.train_pointer += 1

        qa_pairs, act_list, emotion_list = \
            self.dialogues_acts_emotions_into_qas(raw_data, emotion_raw_data, act_raw_data)

        qa_pairs, act_list, emotion_list = np.asarray(qa_pairs), np.asarray(act_list), np.asarray(emotion_list)

        if qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None])
        x_batch = qa_pairs[:, 0]
        y_batch = qa_pairs[:, -1]

        x_act_batch = act_list[:, 0]
        y_act_batch = act_list[:, -1]

        x_emotion_batch = emotion_list[:, 0]
        y_emotion_batch = emotion_list[:, -1]

        x_length = [len(item) for item in x_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), np.asarray(x_length), \
               np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length), \
               np.asarray(x_act_batch), np.asarray(y_act_batch),  \
               np.asarray(x_emotion_batch), np.asarray(y_emotion_batch)

    def get_validation(self):
        if self.val_pointer < self.val_batch_num:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
            act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
            emotion_raw_data = self.validation_emotion_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
        else:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size: ]
            act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size: ]
            emotion_raw_data = self.validation_emotion_raw_text[self.val_pointer * self.val_batch_size: ]

        self.val_pointer += 1

        qa_pairs, act_list, emotion_list = \
            self.dialogues_acts_emotions_into_qas(raw_data, emotion_raw_data, act_raw_data)

        qa_pairs, act_list, emotion_list = np.asarray(qa_pairs), np.asarray(act_list), np.asarray(emotion_list)

        if qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None])
        x_batch = qa_pairs[:, 0]
        y_batch = qa_pairs[:, -1]

        x_act_batch = act_list[:, 0]
        y_act_batch = act_list[:, -1]

        x_emotion_batch = emotion_list[:, 0]
        y_emotion_batch = emotion_list[:, -1]

        x_length = [len(item) for item in x_batch]

        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), np.asarray(x_length), \
               np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length), \
               np.asarray(x_act_batch), np.asarray(y_act_batch),  \
               np.asarray(x_emotion_batch), np.asarray(y_emotion_batch)

    def go_pad(self, sentences, max_length):
        return self.pad_sentence(self.add_go(sentences), max_length)

    def eos_pad(self, sentences, max_length):
        return self.pad_sentence(self.add_eos(sentences), max_length)

    def add_eos(self, sentences):
        eos_sentences = []
        for sentence in sentences:
            new_sentence = copy.copy(sentence)
            new_sentence.append(EOS_ID)
            eos_sentences.append(new_sentence)
        return eos_sentences

    def add_go(self, sentences):
        go_sentences = []
        for sentence in sentences:
            new_sentence = copy.copy(sentence)
            new_sentence.insert(0, GO_ID)
            go_sentences.append(new_sentence)
        return go_sentences

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
        self.num_layers = num_layers

        self.act_class_num = 4
        self.emotion_class_num = 7

        self.create_model()

    def create_model(self):
        self.encoder_input = tf.placeholder(tf.int32, [None, None], name='encoder_input')
        self.encoder_input_lengths = tf.placeholder(tf.int32, [None], name='encoder_input_lengths')
        self.dropout_kp = tf.placeholder(tf.float32, name='dropout_kp')
        # GO
        self.decoder_input = tf.placeholder(tf.int32, [None, None], name='decoder_input')
        # EOS
        self.decoder_target = tf.placeholder(tf.int32, [None, None], name='decoder_target')
        self.decoder_input_lengths = tf.placeholder(tf.int32, [None], name='decoder_input_lengths')
        self.max_sequence_length = tf.reduce_max(self.decoder_input_lengths, name='max_sequence_length')

        self.act_rewards = tf.placeholder(tf.float32, [None, None], name='act_rewards')
        self.emotion_rewards = tf.placeholder(tf.float32, [None, None], name='emotion_rewards')


        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.constant(0., shape=[self.vocab_size, self.embedding_size]), name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size],
                                                        name='embedding_placeholder')
            embeding_init = W.assign(self.embedding_placeholder)
            encoder_embedded_inputs = tf.nn.embedding_lookup(embeding_init, self.encoder_input)
            decoder_embedded_input = tf.nn.embedding_lookup(embeding_init, self.decoder_input)

        with tf.variable_scope('encoder'):
            encoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                encoder_cells.append(encoder_wraped_cell)

            encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                    inputs=encoder_embedded_inputs, dtype=tf.float32,
                                    sequence_length=self.encoder_input_lengths)

        with tf.variable_scope("decoder") as decoder:

            decoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                decoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                decoder_cells.append(decoder_wraped_cell)

            decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embedded_input,
                                                                sequence_length=self.decoder_input_lengths,
                                                                time_major=False)

            output_layer = Dense(self.vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            # 构造decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                               training_helper,
                                                               self.encoder_state,
                                                               output_layer)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                           impute_finished=True,
                                                    maximum_iterations=self.max_sequence_length)

        with tf.variable_scope(decoder, reuse=True):
            start_tokens = tf.tile(tf.constant([GO_ID], dtype=tf.int32), [tf.shape(self.encoder_outputs)[0]])

            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeding_init,
                                                                         start_tokens, EOS_ID)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predicting_helper,
                                                                 self.encoder_state, output_layer)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                impute_finished=False,
                                                            maximum_iterations=self.max_sequence_length)


        self.training_logits = tf.identity(training_decoder_output.rnn_output, name='training_logits')

        self.training_pred_ids = tf.identity(training_decoder_output.sample_id, name='training_pred_ids')

        self.predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predicting_logits')

        masks = tf.sequence_mask(self.decoder_input_lengths, self.max_sequence_length, dtype=tf.float32, name='masks')

        self.all_variables = tf.trainable_variables()

        self.teacher_forcing_cost = tf.reduce_mean(
            tf.contrib.seq2seq.sequence_loss(self.training_logits, self.decoder_target, masks,
                average_across_timesteps=False, average_across_batch=False)
            * self.act_rewards * self.emotion_rewards
        )

        with tf.device('/cpu:0'):
            teacher_forcing_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            teacher_forcing_gradients = teacher_forcing_optimizer.compute_gradients(self.teacher_forcing_cost, self.all_variables)
            capped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var)
                                for grad, var in teacher_forcing_gradients if grad is not None]
            self.teacher_forcing_train_op = teacher_forcing_optimizer.apply_gradients(capped_gradients)

    def get_training_pred_ids(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target):
        training_pred_ids = sess.run(self.training_pred_ids,
                                   feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return training_pred_ids

    def train(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target, act_rewards, emotion_rewards):
        _, loss = sess.run([self.teacher_forcing_train_op, self.teacher_forcing_cost],
                                   feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target,
                                              self.act_rewards: act_rewards,
                                              self.emotion_rewards: emotion_rewards})
        return loss

    def validation(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target, act_rewards, emotion_rewards):
        loss = sess.run(self.teacher_forcing_cost,
                                   feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target,
                                              self.act_rewards: act_rewards,
                                              self.emotion_rewards: emotion_rewards})
        return loss


    def visualization(self, sess, merged, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target, act_labels, emotion_labels):
        loss = sess.run(merged, feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return loss

    def get_train_logit(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target, act_labels, emotion_labels):
        logits = sess.run(self.training_logits,
                                   feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return logits


import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug


MAX_TO_KEEP = 50

EPOCH_SIZE = 50

prefix = 'double_dan_seq2seq'

act_prefix = 'attention_dan_64'
emotion_prefix = 'attention_dan_64_emotion'

def main_train(is_toy=False):
    data_loader = DataLoader(is_toy)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    print('train')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
        model_index = '1'
    else:
        data_loader.load_embedding()
        model_index = '50'
    print('load the embedding matrix')

    act_graph = tf.Graph()

    def get_act_weights(pad_sen_batch, sen_length, act_label, embedding_matrix=data_loader.embedding_matrix,
                            model_index='1', graph=act_graph):
        with graph.as_default():
            with tf.Session(config=config) as act_sess:
                checkpoint_file = 'models_' + act_prefix + '/model-' + model_index

                loader = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                loader.restore(act_sess, checkpoint_file)
                # print('dis-act Model has been restored')

                act_sen_input = graph.get_tensor_by_name('sen_input:0')
                act_sen_input_lengths = graph.get_tensor_by_name('sen_input_lengths:0')
                act_class_label = graph.get_tensor_by_name('class_label:0')
                act_weights_for_reward = graph.get_tensor_by_name('weights_for_reward:0')
                act_dropout_kp = graph.get_tensor_by_name('dropout_kp:0')
                act_accuracy = graph.get_tensor_by_name('accuracy:0')
                act_embedding_placeholder = graph.get_tensor_by_name("embedding/embedding_placeholder:0")

                act_acc, act_weights = act_sess.run([act_accuracy, act_weights_for_reward],
                                           feed_dict={act_sen_input: pad_sen_batch,
                                                      act_sen_input_lengths: sen_length,
                                                      act_class_label: act_label,
                                                      act_embedding_placeholder: embedding_matrix,
                                                      act_dropout_kp: 1.0})
                final_weights = []

                for idx, acc in enumerate(act_acc):
                    if acc > 0.5:
                        final_weights.append(act_weights[idx])
                    else:
                        final_weights.append(1 - act_weights[idx])

                return np.asarray(final_weights)

    emotion_graph = tf.Graph()
    def get_emotion_weights(pad_sen_batch, sen_length, emotion_label, embedding_matrix=data_loader.embedding_matrix,
                            model_index='1', graph=emotion_graph):
        with graph.as_default():
            with tf.Session(config=config) as emotion_sess:
                checkpoint_file = 'models_' + emotion_prefix + '/model-' + model_index

                loader = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                loader.restore(emotion_sess, checkpoint_file)
                # print('dis-emotion Model has been restored')

                emotion_sen_input = graph.get_tensor_by_name('sen_input:0')
                emotion_sen_input_lengths = graph.get_tensor_by_name('sen_input_lengths:0')
                emotion_class_label = graph.get_tensor_by_name('class_label:0')
                emotion_weights_for_reward = graph.get_tensor_by_name('weights_for_reward:0')
                emotion_dropout_kp = graph.get_tensor_by_name('dropout_kp:0')
                emotion_accuracy = graph.get_tensor_by_name('accuracy:0')
                emotion_embedding_placeholder = graph.get_tensor_by_name("embedding/embedding_placeholder:0")

                emotion_acc, emotion_weights = emotion_sess.run([emotion_accuracy, emotion_weights_for_reward],
                                           feed_dict={emotion_sen_input: pad_sen_batch,
                                                      emotion_sen_input_lengths: sen_length,
                                                      emotion_class_label: emotion_label,
                                                      emotion_embedding_placeholder: embedding_matrix,
                                                      emotion_dropout_kp: 1.0})
                final_weights = []

                for idx, acc in enumerate(emotion_acc):
                    if acc > 0.5:
                        final_weights.append(emotion_weights[idx])
                    else:
                        final_weights.append(1 - emotion_weights[idx])

                return np.asarray(final_weights)

    checkpoint_storage = 'models_' + prefix + '/checkpoint'
    checkpoint_dir = 'models_' + prefix + '/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    log_file = checkpoint_dir + '/log.txt'
    log = codecs.open(log_file, 'w')

    double_dan_seq_graph = tf.Graph()
    with double_dan_seq_graph.as_default():
        model = Seq2seq()
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

            # train
            for epoch in range(EPOCH_SIZE):
                losses = 0
                step = 0
                val_losses = 0
                val_step = 0

                for _ in range(data_loader.batch_num + 1):
                    pad_x_batch, x_length, eos_pad_y_batch,  go_pad_y_batch, y_length, \
                        x_act_batch, y_act_batch, x_emotion_batch, y_emotion_batch = data_loader.get_batch_data()
                    if pad_x_batch.all() == None:
                        continue
                    step += 1
                    training_pred_ids = model.get_training_pred_ids(sess,
                                                    pad_x_batch, x_length, go_pad_y_batch,
                                            y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch)

                    tpi_lengths = [len(item) for item in training_pred_ids]

                    tpi_lengths = np.asarray(tpi_lengths)

                    act_weights = get_act_weights(training_pred_ids, tpi_lengths, y_act_batch,
                                                  model_index=model_index)

                    emotion_weights = get_emotion_weights(training_pred_ids, tpi_lengths, y_emotion_batch,
                                                          model_index=model_index)

                    loss_mean = model.train(sess, pad_x_batch, x_length, go_pad_y_batch,
                                            y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch,
                                            act_weights, emotion_weights)

                    losses += loss_mean

                    # if step % 2 == 0:
                    #     result = model.visualization(sess, merged, pad_q_batch, q_length, pad_kb_batch, kb_length, go_pad_y_batch,
                    #                         y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch)
                    #     writer.add_summary(result, step)

                for _ in range(data_loader.val_batch_num + 1):
                    pad_x_batch, x_length, eos_pad_y_batch,  go_pad_y_batch, y_length, \
                        x_act_batch, y_act_batch, x_emotion_batch, y_emotion_batch = data_loader.get_validation()
                    if pad_x_batch.all() == None:
                        continue
                    val_step += 1

                    training_pred_ids = model.get_training_pred_ids(sess,
                                                    pad_x_batch, x_length, go_pad_y_batch,
                                            y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch)

                    tpi_lengths = [len(item) for item in training_pred_ids]

                    tpi_lengths = np.asarray(tpi_lengths)

                    act_weights = get_act_weights(training_pred_ids, tpi_lengths, y_act_batch,
                                                  model_index=model_index)

                    emotion_weights = get_emotion_weights(training_pred_ids, tpi_lengths,
                                                          y_emotion_batch, model_index=model_index)

                    val_loss_mean = model.validation(sess, pad_x_batch, x_length, go_pad_y_batch,
                                            y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch,
                                            act_weights, emotion_weights)

                    val_losses += val_loss_mean

                print('step', step)
                print('val_step', val_step)

                print("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}".format(epoch + 1,
                                            EPOCH_SIZE, losses / step, val_losses / val_step))
                log.write("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}\n".format(epoch + 1,
                                            EPOCH_SIZE, losses / step, val_losses / val_step))

                saver.save(sess, checkpoint_prefix, global_step=epoch + 1)
                print('Model Trained and Saved in epoch ', epoch + 1)

                data_loader.reset_pointer()

    log.close()


if platform.system() == 'Windows':
    from yhd.bleu import *
    from yhd.perplexity import *
else:
    from bleu import *
    from perplexity import *

def main_test(is_toy=False):
    data_loader = DataLoader(is_toy)

    res_file = 'models_' + prefix + '/results.txt'
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

            encoder_input = test_graph.get_tensor_by_name('encoder_input:0')
            encoder_input_lengths = test_graph.get_tensor_by_name('encoder_input_lengths:0')
            dropout_kp = test_graph.get_tensor_by_name('dropout_kp:0')
            decoder_input = test_graph.get_tensor_by_name('decoder_input:0')
            decoder_target = test_graph.get_tensor_by_name('decoder_target:0')
            decoder_input_lengths = test_graph.get_tensor_by_name('decoder_input_lengths:0')
            predicting_logits = test_graph.get_tensor_by_name('predicting_logits:0')
            embedding_placeholder = test_graph.get_tensor_by_name("embedding/embedding_placeholder:0")

            all_test_reply = []

            for _ in range(data_loader.test_batch_num + 1):
                pad_x_batch, x_length, \
                    eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_batch_test()
                if pad_x_batch.all() == None:
                    continue
                predicting_id = sess.run(predicting_logits,
                                           feed_dict={encoder_input: pad_x_batch,
                                                      encoder_input_lengths: x_length,
                                                      decoder_input: go_pad_y_batch,
                                                      decoder_input_lengths: y_length,
                                                      embedding_placeholder: data_loader.embedding_matrix,
                                                      dropout_kp: 1.0,
                                                      decoder_target: eos_pad_y_batch})


                all_reply = []
                for response in predicting_id:
                    all_reply.append([data_loader.id_vocab[id_word]
                                      for id_word in response if id_word != PAD_ID])

                all_test_reply.extend(all_reply)

                print('=========================================================')
                res.write('=========================================================\n')

                for i in range(len(data_loader.test_x_batch)):
                    print('Question:')
                    res.write('Question:\n')
                    print(data_loader.test_x_batch[i])
                    res.write(data_loader.test_x_batch[i] + '\n')
                    print('Answer:')
                    res.write('Answer:\n')
                    print(data_loader.test_y_batch[i])
                    res.write(data_loader.test_y_batch[i] + '\n')
                    print('Generation:')
                    res.write('Generation:\n')
                    print(all_reply[i])
                    res.write(' '.join(all_reply[i]) + '\n')

                    print('---------------------------------------------')
                    res.write('---------------------------------------------\n')

            ppl_input_1 = []
            for line in all_test_reply:
                ppl_input_1.append(' '.join(line))

            bleu, precisions, bp, ratio, translation_length, reference_length \
                    = compute_bleu([data_loader.test_raw_text], all_test_reply, max_order=4)
            print('bleu : ', precisions)
            res.write('bleu : ' + str(precisions) + '\n')

            new_ff_test, ff_trans_cat_table, ff_train_dict = processor(
                ppl_input_1,
                data_loader.source_test
            )
            ppl = calculate_perplexity(new_ff_test, ff_trans_cat_table, ff_train_dict)
            print('perplexity : ', ppl)
            res.write('perplexity : ' + str(ppl) + '\n')

            res.close()


if __name__ == '__main__':
    # main_train(True)

    # main_test(True)

    main_train()

    # main_test()
