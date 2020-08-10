# coding=utf-8
__author__ = 'yhd'

import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F

import platform

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

ASIDE_VOCAB = [PAD_ID, GO_ID, EOS_ID, UNK_ID]

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")

import codecs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader(object):

    def __init__(self, is_toy=False):
        self.is_toy = is_toy
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
            self.source_train_act = 'data_root/dialogues_act_train.txt'
            self.source_val_act = 'data_root/dialogues_act_validation.txt'
            self.source_test_act = 'data_root/dialogues_act_test.txt'

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
      if os.path.exists(vocabulary_path):
        rev_vocab = []

        with codecs.open(vocabulary_path, mode="r", encoding='utf-8') as f:
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
        if self.is_toy:
            self.embedding_matrix = torch.FloatTensor(lookup_table)
        else:
            self.embedding_matrix = torch.cuda.FloatTensor(lookup_table)


    def dialogues_into_qas(self, dialogues):
        qa_pairs = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')[:-1]
            if len(sentences) >= 3:
                for i in range(len(sentences) - 2):
                    # qa = [sentences[i], sentences[i + 1]]
                    qa_pairs.append([self.sentence_to_token_ids(sentences[i]),
                                          self.sentence_to_token_ids(sentences[i + 1]),
                                          self.sentence_to_token_ids(sentences[i + 2])])
        return qa_pairs

    def dialogues_acts_into_qas(self, dialogues, dia_acts):
        qa_pairs = []
        act_list = []
        for idx, dialogue in enumerate(dialogues):
            sentences = dialogue.split('__eou__')[:-1]
            if len(sentences) >= 3:
                for i in range(len(sentences) - 2):
                    # qa = [sentences[i], sentences[i + 1]]
                    qa_pairs.append([self.sentence_to_token_ids(sentences[i]),
                                          self.sentence_to_token_ids(sentences[i + 1]),
                                          self.sentence_to_token_ids(sentences[i + 2])])


            acts = dia_acts[idx].split()
            if len(acts) >= 3:
                for i in range(len(acts) - 2):
                    act_list.append([int(acts[i]) - 1, int(acts[i + 1]) - 1, int(acts[i + 2]) - 1])
        return qa_pairs, act_list

    def dialogues_into_qas_without_id(self, dialogues):
        qa_pairs = []
        for idx, dialogue in enumerate(dialogues):
            sentences = dialogue.split('__eou__')[:-1]
            if len(sentences) >= 3:
                for i in range(len(sentences) - 2):
                    # qa = [sentences[i], sentences[i + 1]]
                    qa_pairs.append([sentences[i], sentences[i + 1], sentences[i + 2]])
        return qa_pairs


    def get_hred_case(self, first, second, reply):

        qa_pairs = [self.sentence_to_token_ids(first),
                    self.sentence_to_token_ids(second),
                    self.sentence_to_token_ids(reply)]

        u1_batch = [qa_pairs[0]]
        u2_batch = [qa_pairs[1]]
        y_batch = [qa_pairs[-1]]

        u1_length = [len(item) for item in u1_batch]
        u2_length = [len(item) for item in u2_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]
        y_length = np.asarray(y_length)

        y_max_length = np.amax(y_length)

        mask = []
        for sen in y_batch:
            m = []
            for word in sen:
                m.append(1)
            m.append(1)
            while len(m) < y_max_length:
                m.append(0)
            mask.append(m)

        return np.asarray(self.pad_sentence(u1_batch, np.amax(u1_length))), np.asarray(u1_length), \
                np.asarray(self.pad_sentence(u2_batch, np.amax(u2_length))), np.asarray(u2_length), \
               np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length)


    def get_batch_test(self):
        if self.test_pointer < self.test_batch_num:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
        else:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size: ]

        self.test_pointer += 1

        qa_pairs = np.asarray(self.dialogues_into_qas(raw_data))
        if qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]),\
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                    np.asarray([None]), np.asarray([None])

        self.test_raw_data = np.asarray(self.dialogues_into_qas_without_id(raw_data))
        self.test_u1_batch = self.test_raw_data[:, 0]
        self.test_u2_batch = self.test_raw_data[:, 1]
        self.test_y_batch = self.test_raw_data[:, -1]

        u1_batch = qa_pairs[:, 0]
        u2_batch = qa_pairs[:, 1]
        y_batch = qa_pairs[:, -1]

        u1_length = [len(item) for item in u1_batch]
        u2_length = [len(item) for item in u2_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]
        y_length = np.asarray(y_length)

        y_max_length = np.amax(y_length)

        mask = []
        for sen in y_batch:
            m = []
            for word in sen:
                m.append(1)
            m.append(1)
            while len(m) < y_max_length:
                m.append(0)
            mask.append(m)

        return np.asarray(self.pad_sentence(u1_batch, np.amax(u1_length))), np.asarray(u1_length), \
                np.asarray(self.pad_sentence(u2_batch, np.amax(u2_length))), np.asarray(u2_length), \
               np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length)

    def get_batch_act_test(self):
        if self.test_pointer < self.test_batch_num:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
            act_raw_data = self.test_act_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
        else:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size: ]
            act_raw_data = self.test_act_raw_text[self.test_pointer * self.test_batch_size: ]

        self.test_pointer += 1

        qa_pairs, act_list = \
            self.dialogues_acts_into_qas(raw_data, act_raw_data)

        qa_pairs, act_list = np.asarray(qa_pairs), np.asarray(act_list)

        if qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                    np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None])

        u1_batch = qa_pairs[:, 0]
        u2_batch = qa_pairs[:, 1]
        y_batch = qa_pairs[:, -1]

        u1_act_batch = act_list[:, 0]
        u2_act_batch = act_list[:, 1]
        y_act_batch = act_list[:, -1]

        u1_length = [len(item) for item in u1_batch]
        u2_length = [len(item) for item in u2_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]
        y_length = np.asarray(y_length)

        y_max_length = np.amax(y_length)

        mask = []
        for sen in y_batch:
            m = []
            for word in sen:
                m.append(1)
            m.append(1)
            while len(m) < y_max_length:
                m.append(0)
            mask.append(m)

        return np.asarray(self.pad_sentence(u1_batch, np.amax(u1_length))), np.asarray(u1_length), \
                np.asarray(self.pad_sentence(u2_batch, np.amax(u2_length))), np.asarray(u2_length), \
               np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length), \
               np.asarray(u1_act_batch), np.asarray(u2_act_batch), np.asarray(y_act_batch)

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

        qa_pairs, act_list = \
            self.dialogues_acts_into_qas(raw_data, act_raw_data)

        qa_pairs, act_list = np.asarray(qa_pairs), np.asarray(act_list)

        if qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                    np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None])
        u1_batch = qa_pairs[:, 0]
        u2_batch = qa_pairs[:, 1]
        y_batch = qa_pairs[:, -1]

        u1_act_batch = act_list[:, 0]
        u2_act_batch = act_list[:, 1]
        y_act_batch = act_list[:, -1]

        u1_length = [len(item) for item in u1_batch]
        u2_length = [len(item) for item in u2_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]
        y_length = np.asarray(y_length)

        y_max_length = np.amax(y_length)

        mask = []
        for sen in y_batch:
            m = []
            for word in sen:
                m.append(1)
            m.append(1)
            while len(m) < y_max_length:
                m.append(0)
            mask.append(m)

        return np.asarray(self.pad_sentence(u1_batch, np.amax(u1_length))), np.asarray(u1_length), \
                np.asarray(self.pad_sentence(u2_batch, np.amax(u2_length))), np.asarray(u2_length), \
               np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length), \
               np.asarray(u1_act_batch), np.asarray(u2_act_batch), np.asarray(y_act_batch), np.asarray(mask)

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

        qa_pairs, act_list = \
            self.dialogues_acts_into_qas(raw_data, act_raw_data)

        qa_pairs, act_list = np.asarray(qa_pairs), np.asarray(act_list)

        if qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                    np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None])
        u1_batch = qa_pairs[:, 0]
        u2_batch = qa_pairs[:, 1]
        y_batch = qa_pairs[:, -1]

        u1_act_batch = act_list[:, 0]
        u2_act_batch = act_list[:, 1]
        y_act_batch = act_list[:, -1]

        u1_length = [len(item) for item in u1_batch]
        u2_length = [len(item) for item in u2_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]
        y_length = np.asarray(y_length)

        y_max_length = np.amax(y_length)

        mask = []
        for sen in y_batch:
            m = []
            for word in sen:
                m.append(1)
            m.append(1)
            while len(m) < y_max_length:
                m.append(0)
            mask.append(m)

        return np.asarray(self.pad_sentence(u1_batch, np.amax(u1_length))), np.asarray(u1_length), \
                np.asarray(self.pad_sentence(u2_batch, np.amax(u2_length))), np.asarray(u2_length), \
               np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length), \
               np.asarray(u1_act_batch), np.asarray(u2_act_batch), np.asarray(y_act_batch), np.asarray(mask)

    def batch_acts_into_items(self, dialogues, dia_acts):
        sen_list = []
        act_list = []
        for idx, ele in enumerate(dia_acts):
            sen_acts = ele.split()
            sentences = dialogues[idx].split('__eou__')[:-1]
            for jdx, item in enumerate(sen_acts):
                sen_list.append(self.sentence_to_token_ids(sentences[jdx]))
                act_list.append(int(item) -  1)
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

        sen_batch, class_batch = self.batch_acts_into_items(raw_data, act_raw_data)
        sen_batch, class_batch = np.asarray(sen_batch), np.asarray(class_batch)
        if sen_batch.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None])

        sen_length = [len(item) for item in sen_batch]

        return np.asarray(self.pad_sentence(sen_batch, np.amax(sen_length))), np.asarray(sen_length), \
                np.asarray(class_batch)

    def get_act_validation(self):
        if self.val_pointer < self.val_batch_num:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
            act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
        else:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size: ]
            act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size: ]

        self.val_pointer += 1

        sen_batch, class_batch = self.batch_acts_into_items(raw_data, act_raw_data)
        sen_batch, class_batch = np.asarray(sen_batch), np.asarray(class_batch)
        if sen_batch.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None])

        sen_length = [len(item) for item in sen_batch]

        return np.asarray(self.pad_sentence(sen_batch, np.amax(sen_length))), np.asarray(sen_length), \
                np.asarray(class_batch)


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

# class DAN_ACT(nn.Module):
#
#     def __init__(self, embeddings):
#         super(DAN_ACT, self).__init__()
#         self.embedding_size = EMBEDDING_SIZE
#         layer1_units = 100
#         self.act_class_num = 4
#         self.drop_out = 0.2
#
#         self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)
#
#         self.layers = nn.Sequential(nn.Linear(self.embedding_size, layer1_units),
#                                             nn.Dropout(self.drop_out),
#                                             nn.Linear(layer1_units, self.act_class_num),
#                                             nn.Dropout(self.drop_out))
#
#
#     def forward(self, x_batch, is_test=False):
#         embed = self.embedding_layer(x_batch)
#         reward = self.layers(embed)
#         output = torch.mean(reward, 1)
#
#         if is_test:
#             output, reward = F.softmax(output, dim=1), F.softmax(reward, dim=2)
#         return output, reward


class Generator(nn.Module):
    def __init__(self, embeddings):
        super(Generator, self).__init__()
        self.hidden_size = EMBEDDING_SIZE
        self.output_size = VOCAB_SIZE

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.encoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.session_encoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.docoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def transfer_index(self, a_sort):
        a_l = a_sort.tolist()
        m_l = []
        for i in range(len(a_l)):
            m_l.append(a_l.index(i))
        return m_l

    def reverse_sort(self, u_input, u_lens):
        u_sort = np.argsort(u_lens * -1)
        u_sort_input = u_input[u_sort, :]
        u_sort_lens = np.sort(u_lens * -1) * -1

        u_embed = self.embedding_layer(u_sort_input)
        u_packed = torch.nn.utils.rnn.pack_padded_sequence(u_embed, u_sort_lens,
                                                                 batch_first=True)

        encoder_outputs, encoder_hidden = self.encoder(u_packed, None)
        encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs,
                                                                    batch_first=True,
                                                                    padding_value=PAD_ID)

        m_sort = self.transfer_index(u_sort)
        u_outputs = encoder_outputs[m_sort, :]
        u_hidden = encoder_hidden.squeeze(0)[m_sort]
        return u_outputs, u_hidden


    def forward(self, u1_input, u1_lens, u2_input, u2_lens, target_input, target_input_lens, is_test=False):

        u1_outputs, u1_hidden = self.reverse_sort(u1_input, u1_lens)

        u2_outputs, u2_hidden = self.reverse_sort(u2_input, u2_lens)

        si = torch.stack((u1_hidden, u2_hidden), 1)

        print(si)

        if is_test:
            session_input = si
        else:
            session_input = si.squeeze()

        session_outputs, session_hidden = self.session_encoder(session_input, None)

        target_embed = self.embedding_layer(target_input)
        # target_packed = torch.nn.utils.rnn.pack_padded_sequence(target_embed, target_input_lens,
        #                                                          batch_first=True)

        decoder_hidden = session_hidden
        batch_size = target_input.shape[0]
        target_max_seq_lens = target_input.shape[1]

        if is_test:
            start = torch.cuda.LongTensor([[GO_ID for _ in range(batch_size)]])
            emb = self.embedding_layer(start)
        outputs = []
        for idx in range(target_max_seq_lens):
            if is_test:
                decoder_output, decoder_hidden = self.docoder(emb.view(batch_size, 1, -1), decoder_hidden)
                output_vocab = self.output_layer(decoder_output)
                output_id = torch.argmax(output_vocab, dim=2)
                emb = self.embedding_layer(output_id)
                # if output_id.item() == EOS_ID:
                #     break
                outputs.append(output_vocab)
            else:
                decoder_output, decoder_hidden = self.docoder(target_embed[:, idx, :].view(batch_size, 1, -1),
                                                      decoder_hidden)
                outputs.append(self.output_layer(decoder_output))
        return outputs

def maskNLLLoss(loss_func, inp, target, mask, rewards=None):
    nTotal = mask.sum()
    ce = loss_func(inp, target)
    if rewards is not None:
        ce_rewards = ce * rewards
        loss = ce_rewards.masked_select(mask).mean()
    else:
        loss = ce.masked_select(mask).mean()
    return loss, nTotal.item()

EPOCH_SIZE = 50

prefix = 'hred_torch'

act_prefix = 'models_simple_dan/adan_para_%s.pkl'

model_prefix = 'models_' + prefix + '/generator_para_%s.pkl'


def main_train(is_toy=False):

    # data
    data_loader = DataLoader(is_toy)
    print('train')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    else:
        data_loader.load_embedding()
    print('load the embedding matrix')

    is_toy = False

    # # model

    # if is_toy:
    #     act = DAN_ACT(data_loader.embedding_matrix)
    #     act.load_state_dict(torch.load(act_prefix%'10000'))
    # else:
    #     act = DAN_ACT(data_loader.embedding_matrix).cuda()
    #     act.load_state_dict(torch.load(act_prefix%'50'))

    if is_toy:
        generator = Generator(data_loader.embedding_matrix)
        # generator.load_state_dict(torch.load(model_prefix%'2'))
    else:
        generator = Generator(data_loader.embedding_matrix).cuda()
        # generator.load_state_dict(torch.load(model_prefix%'3'))
        # print('Model 3 has been loaded')
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    # generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.00001)

    loss_func = nn.CrossEntropyLoss()

    checkpoint_dir = 'models_' + prefix + '/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    log_file = checkpoint_dir + 'log.txt'
    log = codecs.open(log_file, 'a')


    # train
    for epoch in range(EPOCH_SIZE):

        losses = 0
        step = 0
        val_losses = 0
        val_step = 0

        generator.train()

        for bn in range(data_loader.batch_num + 1):
            pad_u1_batch, u1_length, pad_u2_batch, u2_length, eos_pad_y_batch, \
                go_pad_y_batch, y_length, \
                    u1_act_batch, u2_act_batch, y_act_batch, y_mask = data_loader.get_batch_data()

            if pad_u1_batch.all() == None:
                continue

            # 18
            batch_size = eos_pad_y_batch.shape[0]
            # 30
            y_seq_len = eos_pad_y_batch.shape[1]

            loss_mean = 0

            if is_toy:
                pad_u1, pad_u2, eos_pad_y, go_pad_y, u1_length_, u2_length_, y_length_, y_mask = \
                    torch.LongTensor(pad_u1_batch), \
                    torch.LongTensor(pad_u2_batch), \
                    torch.LongTensor(eos_pad_y_batch), \
                    torch.LongTensor(go_pad_y_batch), \
                    torch.LongTensor(u1_length), \
                    torch.LongTensor(u2_length), \
                    torch.LongTensor(y_length), \
                    torch.ByteTensor(y_mask)
            else:
                pad_u1, pad_u2, eos_pad_y, go_pad_y, u1_length_, u2_length_, y_length_, y_mask = \
                    torch.cuda.LongTensor(pad_u1_batch), \
                    torch.cuda.LongTensor(pad_u2_batch), \
                    torch.cuda.LongTensor(eos_pad_y_batch), \
                    torch.cuda.LongTensor(go_pad_y_batch), \
                    torch.cuda.LongTensor(u1_length), \
                    torch.cuda.LongTensor(u2_length), \
                    torch.cuda.LongTensor(y_length), \
                    torch.cuda.ByteTensor(y_mask)

            step += 1

            outputs = generator(pad_u1, u1_length_, pad_u2, u2_length_, go_pad_y, y_length_)
            # outputs = torch.squeeze(torch.stack(outputs, 0))
            # outputs_ids = torch.transpose(outputs, 0, 1)
            # outputs_ids = torch.argmax(outputs_ids, dim=2)
            #
            # # act
            # act_outputs, act_rewards = act(outputs_ids, True)
            #
            # final_act_rewards = []
            # for i in range(batch_size):
            #     act_rew = []
            #     for j in range(y_seq_len):
            #         act_rew.append(act_rewards[i][j][y_act_batch[i]])
            #     final_act_rewards.append(act_rew)
            #
            # final_act_rewards = torch.cuda.FloatTensor(final_act_rewards)

            n_total = 0

            for jdx in range(y_seq_len):
                # mask_loss, nTotal = maskNLLLoss(loss_func, outputs[jdx].view(batch_size, -1),
                #                                 eos_pad_y[:, jdx], y_mask[:, jdx],
                #                                 final_act_rewards[:, jdx])
                mask_loss, nTotal = maskNLLLoss(loss_func, outputs[jdx].view(batch_size, -1),
                                                eos_pad_y[:, jdx], y_mask[:, jdx])
                loss_mean += mask_loss
                n_total += nTotal

            generator_optimizer.zero_grad()
            loss_mean.backward()
            generator_optimizer.step()

            losses += loss_mean

        generator.eval()

        for _ in range(data_loader.val_batch_num + 1):
            pad_u1_batch, u1_length, pad_u2_batch, u2_length, eos_pad_y_batch, \
                go_pad_y_batch, y_length, \
                    u1_act_batch, u2_act_batch, y_act_batch, y_mask = data_loader.get_validation()


            if pad_u1_batch.all() == None:
                continue

            batch_size = eos_pad_y_batch.shape[0]
            y_seq_len = eos_pad_y_batch.shape[1]

            val_loss_mean = 0

            if is_toy:
                pad_u1, pad_u2, eos_pad_y, go_pad_y, u1_length_, u2_length_, y_length_, y_mask = \
                    torch.LongTensor(pad_u1_batch), \
                    torch.LongTensor(pad_u2_batch), \
                    torch.LongTensor(eos_pad_y_batch), \
                    torch.LongTensor(go_pad_y_batch), \
                    torch.LongTensor(u1_length), \
                    torch.LongTensor(u2_length), \
                    torch.LongTensor(y_length), \
                    torch.ByteTensor(y_mask)
            else:
                pad_u1, pad_u2, eos_pad_y, go_pad_y, u1_length_, u2_length_, y_length_, y_mask = \
                    torch.cuda.LongTensor(pad_u1_batch), \
                    torch.cuda.LongTensor(pad_u2_batch), \
                    torch.cuda.LongTensor(eos_pad_y_batch), \
                    torch.cuda.LongTensor(go_pad_y_batch), \
                    torch.cuda.LongTensor(u1_length), \
                    torch.cuda.LongTensor(u2_length), \
                    torch.cuda.LongTensor(y_length), \
                    torch.cuda.ByteTensor(y_mask)


            outputs = generator(pad_u1, u1_length_, pad_u2, u2_length_, go_pad_y, y_length_)

            n_total = 0

            for jdx in range(y_seq_len):
                mask_loss, nTotal = maskNLLLoss(loss_func, outputs[jdx].view(batch_size, -1),
                                                eos_pad_y[:, jdx], y_mask[:, jdx])
                val_loss_mean += mask_loss
                n_total += nTotal

            generator_optimizer.zero_grad()

            val_step += 1
            val_losses += torch.mean(val_loss_mean).item() / n_total


        print("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}".format(epoch + 1,
                                    EPOCH_SIZE, losses / step, val_losses / val_step))
        log.write("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}\n".format(epoch + 1,
                                    EPOCH_SIZE, losses / step, val_losses / val_step))

        torch.save(generator.state_dict(), checkpoint_dir + 'generator_para_' + str(epoch + 1) + '.pkl')
        print('Model Trained and Saved in epoch ', epoch + 1)

        data_loader.reset_pointer()

    log.close()



def response_hred(first, second, reply, is_toy=False):
    data_loader = DataLoader(is_toy)

    load_index = '4'

    # test
    print('test')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    else:
        data_loader.load_embedding()
    print('load the embedding matrix')

    is_toy = False

    checkpoint_file = 'models_' + prefix + '/generator_para_%s.pkl'
    if is_toy:
        generator = Generator(data_loader.embedding_matrix)
    else:
        generator = Generator(data_loader.embedding_matrix).cuda()
    generator.load_state_dict(torch.load(checkpoint_file % load_index))
    print('Model has been restored')

    pad_u1_batch, u1_length, pad_u2_batch, u2_length, \
        eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_hred_case(first, second, reply)

    pad_u1_batch_test, u1_length_test, pad_u2_batch_test, u2_length_test, \
            eos_pad_y_batch_test, go_pad_y_batch_test, y_length_test = data_loader.get_batch_test()

    # print(pad_u1_batch.shape, pad_u1_batch_test.shape)
    # print(u1_length.shape, u1_length_test.shape)
    # print(pad_u2_batch.shape, pad_u2_batch_test.shape)
    # print(u2_length.shape, u2_length_test.shape)
    # print(eos_pad_y_batch.shape, eos_pad_y_batch_test.shape)
    # print(go_pad_y_batch.shape, go_pad_y_batch_test.shape)
    # print(y_length.shape, y_length_test.shape)

    # 18
    batch_size = eos_pad_y_batch.shape[0]
    # 30
    y_seq_len = eos_pad_y_batch.shape[1]

    if is_toy:
        pad_u1, pad_u2, eos_pad_y, go_pad_y, u1_length_, u2_length_, y_length_ = \
            torch.LongTensor(pad_u1_batch), \
            torch.LongTensor(pad_u2_batch), \
            torch.LongTensor(eos_pad_y_batch), \
            torch.LongTensor(go_pad_y_batch), \
            torch.LongTensor(u1_length), \
            torch.LongTensor(u2_length), \
            torch.LongTensor(y_length)
    else:
        pad_u1, pad_u2, eos_pad_y, go_pad_y, u1_length_, u2_length_, y_length_ = \
            torch.cuda.LongTensor(pad_u1_batch), \
            torch.cuda.LongTensor(pad_u2_batch), \
            torch.cuda.LongTensor(eos_pad_y_batch), \
            torch.cuda.LongTensor(go_pad_y_batch), \
            torch.cuda.LongTensor(u1_length), \
            torch.cuda.LongTensor(u2_length), \
            torch.cuda.LongTensor(y_length)

    outputs = generator(pad_u1, u1_length_, pad_u2, u2_length_, go_pad_y, y_length_, True)


    all_reply = []
    for i in range(batch_size):
        reply = []
        for j in range(y_seq_len):
            id_word = torch.argmax(outputs[j][i]).item()
            if id_word != PAD_ID and id_word != UNK_ID and id_word != GO_ID and id_word != EOS_ID:
                reply.append(data_loader.id_vocab[id_word])
        all_reply.append(reply)

    print('Generation:')
    print(' '.join(all_reply[0]))


def main_test(is_toy=False):
    data_loader = DataLoader(is_toy)

    res_file = 'models_' + prefix + '/' + prefix + '_4_results.txt'
    res = codecs.open(res_file, 'w')

    reply_file = 'models_' + prefix + '/' + prefix + '_4_reply.txt'
    reply_f = codecs.open(reply_file, 'w')

    ans_file = 'models_' + prefix + '/' + prefix + '_4_answer.txt'
    ans_f = codecs.open(ans_file, 'w')

    load_index = '4'

    # test
    print('test')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    else:
        data_loader.load_embedding()
    print('load the embedding matrix')

    is_toy = False

    checkpoint_file = 'models_' + prefix + '/generator_para_%s.pkl'
    if is_toy:
        generator = Generator(data_loader.embedding_matrix)
    else:
        generator = Generator(data_loader.embedding_matrix).cuda()
    generator.load_state_dict(torch.load(checkpoint_file % load_index))
    print('Model has been restored')

    for _ in range(data_loader.test_batch_num + 1):

        pad_u1_batch, u1_length, pad_u2_batch, u2_length, \
            eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_batch_test()

        if pad_u1_batch.all() == None:
            continue

        # 18
        batch_size = eos_pad_y_batch.shape[0]
        # 30
        y_seq_len = eos_pad_y_batch.shape[1]

        if is_toy:
            pad_u1, pad_u2, eos_pad_y, go_pad_y, u1_length_, u2_length_, y_length_ = \
                torch.LongTensor(pad_u1_batch), \
                torch.LongTensor(pad_u2_batch), \
                torch.LongTensor(eos_pad_y_batch), \
                torch.LongTensor(go_pad_y_batch), \
                torch.LongTensor(u1_length), \
                torch.LongTensor(u2_length), \
                torch.LongTensor(y_length)
        else:
            pad_u1, pad_u2, eos_pad_y, go_pad_y, u1_length_, u2_length_, y_length_ = \
                torch.cuda.LongTensor(pad_u1_batch), \
                torch.cuda.LongTensor(pad_u2_batch), \
                torch.cuda.LongTensor(eos_pad_y_batch), \
                torch.cuda.LongTensor(go_pad_y_batch), \
                torch.cuda.LongTensor(u1_length), \
                torch.cuda.LongTensor(u2_length), \
                torch.cuda.LongTensor(y_length)

        outputs = generator(pad_u1, u1_length_, pad_u2, u2_length_, go_pad_y, y_length_, True)


        all_reply = []
        for i in range(batch_size):
            reply = []
            for j in range(y_seq_len):
                id_word = torch.argmax(outputs[j][i]).item()
                if id_word != PAD_ID and id_word != UNK_ID and id_word != GO_ID and id_word != EOS_ID:
                    reply.append(data_loader.id_vocab[id_word])
            all_reply.append(reply)


        print('=========================================================')
        res.write('=========================================================\n')

        for i in range(len(data_loader.test_u1_batch)):
            print('First Turn:')
            res.write('First Turn:\n')
            print(data_loader.test_u1_batch[i])
            res.write(data_loader.test_u1_batch[i] + '\n')
            print('Second Turn:')
            res.write('Second Turn:\n')
            print(data_loader.test_u2_batch[i])
            res.write(data_loader.test_u2_batch[i] + '\n')
            print('Answer:')
            res.write('Answer:\n')
            print(data_loader.test_y_batch[i])
            res.write(data_loader.test_y_batch[i] + '\n')
            ans_f.write(data_loader.test_y_batch[i] + '\n')
            print('Generation:')
            res.write('Generation:\n')
            print(' '.join(all_reply[i]))
            res.write(' '.join(all_reply[i]) + '\n')
            reply_f.write(' '.join(all_reply[i]) + '\n')

            print('---------------------------------------------')
            res.write('---------------------------------------------\n')

    res.close()
    reply_f.close()
    ans_f.close()

def get_test_act():
    data_loader = DataLoader()

    res_file = 'models_' + prefix + '/' + prefix + '_acts.txt'
    res = codecs.open(res_file, 'w')

    # test
    print('test act')
    data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    print('load the embedding matrix')

    for _ in range(data_loader.test_batch_num + 1):

        pad_u1_batch, u1_length, pad_u2_batch, u2_length, \
            eos_pad_y_batch, go_pad_y_batch, y_length, u1_act_batch, u2_act_batch,\
            y_act_batch = data_loader.get_batch_act_test()

        if pad_u1_batch.all() == None:
            continue

        for ele in y_act_batch:
            res.write(str(ele) + '\n')

    res.close()

if __name__ == '__main__':
    # main_train(True)

    # main_test(True)

    # main_train()

    # main_test()

    response_hred('what for ?', 'i am going to read books', 'fine , ok')


    # get_test_act()

    # d = DataLoader(True)
    #
    # res = d.get_batch_data()
    # print(res)
    #
    # # for i in range(d.batch_num + 1):
    # #     res = d.get_batch_data()
    # #     print(res[4])
    # #     # print(res[2])
    # #     print('----------------------------------------------')


