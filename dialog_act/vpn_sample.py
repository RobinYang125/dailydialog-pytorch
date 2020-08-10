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
            # self.source_train_act = 'data_root/act_train.txt'
            # self.source_val_act = 'data_root/act_val.txt'
            # self.source_test_act = 'data_root/act_test.txt'
            # self.source_train_emotion = 'data_root/emotion_train.txt'
            # self.source_val_emotion = 'data_root/emotion_validation.txt'
            # self.source_test_emotion = 'data_root/emotion_test.txt'
            self.train_nouns = 'data_root/words/train_nouns.txt'
            self.train_verbs = 'data_root/words/train_verbs.txt'
            self.train_prons = 'data_root/words/train_prons.txt'

            self.val_nouns = 'data_root/words/val_nouns.txt'
            self.val_verbs = 'data_root/words/val_verbs.txt'
            self.val_prons = 'data_root/words/val_prons.txt'

            self.test_nouns = 'data_root/words/test_nouns.txt'
            self.test_verbs = 'data_root/words/test_verbs.txt'
            self.test_prons = 'data_root/words/test_prons.txt'

            self.train_tag = 'data_root/words/train_tag.txt'
            self.val_tag = 'data_root/words/val_tag.txt'
            self.test_tag = 'data_root/words/test_tag.txt'
        else:
            self.source_train = 'data_root/dialogues_train.txt'
            self.source_test = 'data_root/dialogues_test.txt'
            self.batch_size = BATCH_SIZE
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = 'data_root/dialogues_validation.txt'
            self.test_batch_size = BATCH_SIZE
            self.val_batch_size = BATCH_SIZE
            # self.source_train_act = 'data_root/dialogues_act_train.txt'
            # self.source_val_act = 'data_root/dialogues_act_validation.txt'
            # self.source_test_act = 'data_root/dialogues_act_test.txt'
            # self.source_train_emotion = 'data_root/dialogues_emotion_train.txt'
            # self.source_val_emotion = 'data_root/dialogues_emotion_validation.txt'
            # self.source_test_emotion = 'data_root/dialogues_emotion_test.txt'

            self.train_nouns = 'data_root/words/dialogues_train_nouns.txt'
            self.train_verbs = 'data_root/words/dialogues_train_verbs.txt'
            self.train_prons = 'data_root/words/dialogues_train_prons.txt'

            self.val_nouns = 'data_root/words/dialogues_validation_nouns.txt'
            self.val_verbs = 'data_root/words/dialogues_validation_verbs.txt'
            self.val_prons = 'data_root/words/dialogues_validation_prons.txt'

            self.test_nouns = 'data_root/words/dialogues_test_nouns.txt'
            self.test_verbs = 'data_root/words/dialogues_test_verbs.txt'
            self.test_prons = 'data_root/words/dialogues_test_prons.txt'

            self.train_tag = 'data_root/words/dialogues_train_tag.txt'
            self.val_tag = 'data_root/words/dialogues_validation_tag.txt'
            self.test_tag = 'data_root/words/dialogues_test_tag.txt'

        if platform.system() == 'Windows':
            with open(self.source_train, 'r', encoding='utf-8') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.train_nouns, 'r', encoding='utf-8') as stf:
                self.train_nouns_raw_text = stf.readlines()
            with open(self.train_verbs, 'r', encoding='utf-8') as stf:
                self.train_verbs_raw_text = stf.readlines()
            with open(self.train_prons, 'r', encoding='utf-8') as stf:
                self.train_prons_raw_text = stf.readlines()
            with open(self.train_tag, 'r', encoding='utf-8') as stf:
                self.train_tag_raw_text = stf.readlines()

            # with open(self.source_train_act, 'r', encoding='utf-8') as stf:
            #     self.train_act_raw_text = stf.readlines()
            #
            # with open(self.source_train_emotion, 'r', encoding='utf-8') as stf:
            #     self.train_emotion_raw_text = stf.readlines()

            with open(self.source_validation, 'r', encoding='utf-8') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.val_nouns, 'r', encoding='utf-8') as stf:
                self.val_nouns_raw_text = stf.readlines()
            with open(self.val_verbs, 'r', encoding='utf-8') as stf:
                self.val_verbs_raw_text = stf.readlines()
            with open(self.val_prons, 'r', encoding='utf-8') as stf:
                self.val_prons_raw_text = stf.readlines()
            with open(self.val_tag, 'r', encoding='utf-8') as stf:
                self.val_tag_raw_text = stf.readlines()

            # with open(self.source_val_act, 'r', encoding='utf-8') as svf:
            #     self.validation_act_raw_text = svf.readlines()
            #
            # with open(self.source_val_emotion, 'r', encoding='utf-8') as svf:
            #     self.validation_emotion_raw_text = svf.readlines()

            with open(self.source_test, 'r', encoding='utf-8') as stef:
                self.test_raw_text = stef.readlines()

            with open(self.test_nouns, 'r', encoding='utf-8') as stf:
                self.test_nouns_raw_text = stf.readlines()
            with open(self.test_verbs, 'r', encoding='utf-8') as stf:
                self.test_verbs_raw_text = stf.readlines()
            with open(self.test_prons, 'r', encoding='utf-8') as stf:
                self.test_prons_raw_text = stf.readlines()
            with open(self.test_tag, 'r', encoding='utf-8') as stf:
                self.test_tag_raw_text = stf.readlines()

            # with open(self.source_test_act, 'r', encoding='utf-8') as stef:
            #     self.test_act_raw_text = stef.readlines()
            #
            # with open(self.source_test_emotion, 'r', encoding='utf-8') as stef:
            #     self.test_emotion_raw_text = stef.readlines()
        else:
            with open(self.source_train, 'r') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.train_nouns, 'r') as stf:
                self.train_nouns_raw_text = stf.readlines()
            with open(self.train_verbs, 'r') as stf:
                self.train_verbs_raw_text = stf.readlines()
            with open(self.train_prons, 'r') as stf:
                self.train_prons_raw_text = stf.readlines()
            with open(self.train_tag, 'r') as stf:
                self.train_tag_raw_text = stf.readlines()

            # with open(self.source_train_act, 'r') as stf:
            #     self.train_act_raw_text = stf.readlines()
            #
            # with open(self.source_train_emotion, 'r') as stf:
            #     self.train_emotion_raw_text = stf.readlines()

            with open(self.source_validation, 'r') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.val_nouns, 'r') as stf:
                self.val_nouns_raw_text = stf.readlines()
            with open(self.val_verbs, 'r') as stf:
                self.val_verbs_raw_text = stf.readlines()
            with open(self.val_prons, 'r') as stf:
                self.val_prons_raw_text = stf.readlines()
            with open(self.val_tag, 'r') as stf:
                self.val_tag_raw_text = stf.readlines()

            # with open(self.source_val_act, 'r') as svf:
            #     self.validation_act_raw_text = svf.readlines()
            #
            # with open(self.source_val_emotion, 'r') as svf:
            #     self.validation_emotion_raw_text = svf.readlines()

            with open(self.source_test, 'r') as stef:
                self.test_raw_text = stef.readlines()

            with open(self.test_nouns, 'r') as stf:
                self.test_nouns_raw_text = stf.readlines()
            with open(self.test_verbs, 'r') as stf:
                self.test_verbs_raw_text = stf.readlines()
            with open(self.test_prons, 'r') as stf:
                self.test_prons_raw_text = stf.readlines()
            with open(self.test_tag, 'r') as stf:
                self.test_tag_raw_text = stf.readlines()

            # with open(self.source_test_act, 'r') as stef:
            #     self.test_act_raw_text = stef.readlines()
            #
            # with open(self.source_test_emotion, 'r') as stef:
            #     self.test_emotion_raw_text = stef.readlines()


        self.batch_num = len(self.train_raw_text) // self.batch_size
        self.val_batch_num = len(self.validation_raw_text) // self.val_batch_size
        self.test_batch_num = len(self.test_raw_text) // self.test_batch_size

        self.train_pointer = 0
        self.val_pointer = 0
        self.test_pointer = 0

        self.initialize_vocabulary()
        self.initialize_nouns_vocabulary()
        self.initialize_verbs_vocabulary()
        self.initialize_prons_vocabulary()
        # self.map_between_vocab()

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

    def initialize_nouns_vocabulary(self, vocabulary_path='data_root/nouns10000.in'):
      if os.path.exists(vocabulary_path):
        rev_vocab = []

        with codecs.open(vocabulary_path, mode="r", encoding='utf-8') as f:
          rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

        self.nouns_vocab_id = vocab
        self.nouns_id_vocab = {v: k for k, v in vocab.items()}
        self.nouns_rev_vocab = rev_vocab

    def initialize_verbs_vocabulary(self, vocabulary_path='data_root/verbs10000.in'):
      if os.path.exists(vocabulary_path):
        rev_vocab = []

        with codecs.open(vocabulary_path, mode="r", encoding='utf-8') as f:
          rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

        self.verbs_vocab_id = vocab
        self.verbs_id_vocab = {v: k for k, v in vocab.items()}
        self.verbs_rev_vocab = rev_vocab

    def initialize_prons_vocabulary(self, vocabulary_path='data_root/prons10000.in'):
      if os.path.exists(vocabulary_path):
        rev_vocab = []

        with codecs.open(vocabulary_path, mode="r", encoding='utf-8') as f:
          rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

        self.prons_vocab_id = vocab
        self.prons_id_vocab = {v: k for k, v in vocab.items()}
        self.prons_rev_vocab = rev_vocab

    # def map_between_vocab(self, source_vocab, target_vocab):


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
            sentences = dialogue.split('__eou__')
            for i in range(len(sentences) - 2):
                qa = [sentences[i], sentences[i + 1]]
                qa_pairs.append([self.sentence_to_token_ids(sentences[i]),
                                      self.sentence_to_token_ids(sentences[i + 1])])
        return qa_pairs

    def dialogues_vpn_into_qas(self, dialogues, verbs, prons, nouns, tags):
        qa_pairs = []
        verbs_list = []
        prons_list = []
        nouns_list = []
        tags_list = []
        for idx, dialogue in enumerate(dialogues):
            sentences = dialogue.split('__eou__')[:-1]
            v_s = verbs[idx].split('__eou__')[:-1]
            p_s = prons[idx].split('__eou__')[:-1]
            n_s = nouns[idx].split('__eou__')[:-1]
            t_s = tags[idx].split('__eou__')[:-1]
            for i in range(len(sentences) - 1):
                if len(v_s[i]) > 0 and len(p_s[i]) > 0 and len(n_s[i]) > 0:
                    qa_pairs.append([self.sentence_to_token_ids(sentences[i]),
                                          self.sentence_to_token_ids(sentences[i + 1])])
                    verbs_list.append(self.sentence_to_token_ids(v_s[i]))
                    prons_list.append(self.sentence_to_token_ids(p_s[i]))
                    nouns_list.append(self.sentence_to_token_ids(n_s[i]))
                    tags_list.append(t_s[i + 1].split())

        return qa_pairs, verbs_list, prons_list, nouns_list, tags_list

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
            verbs_raw_data = self.test_verbs_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
            prons_raw_data = self.test_prons_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
            nouns_raw_data = self.test_nouns_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
            tags_raw_data = self.test_tag_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
        else:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size: ]
            verbs_raw_data = self.test_verbs_raw_text[self.test_pointer * self.test_batch_size: ]
            prons_raw_data = self.test_prons_raw_text[self.test_pointer * self.test_batch_size: ]
            nouns_raw_data = self.test_nouns_raw_text[self.test_pointer * self.test_batch_size: ]
            tags_raw_data = self.test_tag_raw_text[self.test_pointer * self.test_batch_size: ]

        self.test_pointer += 1

        # self.test_raw_data = np.asarray(self.dialogues_into_qas_without_id(raw_data))
        # self.test_x_batch = self.test_raw_data[:, 0]
        # self.test_y_batch = self.test_raw_data[:, -1]

        qa_pairs, verbs_list, prons_list, nouns_list, tags_list = \
            self.dialogues_vpn_into_qas(raw_data, verbs_raw_data, prons_raw_data, nouns_raw_data, tags_raw_data)

        qa_pairs = np.asarray(qa_pairs)

        # qa_pairs, verbs_list, prons_list, nouns_list\
        #     = np.asarray(qa_pairs), np.asarray(verbs_list), np.asarray(prons_list), \
        #                     np.asarray(nouns_list)

        if len(qa_pairs) == 0:
            return np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                    np.asarray([None]), np.asarray([None]), np.asarray([None])
        x_batch = qa_pairs[:, 0]
        y_batch = qa_pairs[:, -1]

        x_v_batch = verbs_list
        x_p_batch = prons_list
        x_n_batch = nouns_list
        y_tag_batch = tags_list

        x_length = [len(item) for item in x_batch]
        x_v_length = [len(item) for item in x_v_batch]
        x_p_length = [len(item) for item in x_p_batch]
        x_n_length = [len(item) for item in x_n_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]


        map_y_batch = []
        for idx, batch in enumerate(y_batch):
            y_tag = y_tag_batch[idx]
            map_item = []
            for jdx, ele in enumerate(batch):
                if y_tag[jdx] == '0':
                    cont = self.nouns_vocab_id.get(self.id_vocab[ele], UNK_ID)
                elif y_tag[jdx] == '1':
                    cont = self.verbs_vocab_id.get(self.id_vocab[ele], UNK_ID)
                elif y_tag[jdx] == '2':
                    cont = self.prons_vocab_id.get(self.id_vocab[ele], UNK_ID)
                else:
                    cont = ele
                map_item.append(cont)
            map_y_batch.append(map_item)


        go_y_batch = self.add_go(y_batch)
        eos_map = self.add_eos_tag(map_y_batch, 3)
        eos_y_tag_batch = self.str_into_int(self.add_eos_tag(y_tag_batch, '3'))

        # y_max_length = np.amax(y_length)

        return x_batch, x_v_batch, x_p_batch, x_n_batch, x_length, x_v_length, x_p_length, \
               x_n_length, go_y_batch, y_length, eos_map, eos_y_tag_batch

    def get_batch_data(self):
        if self.train_pointer < self.batch_num:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
            # act_raw_data = self.train_act_raw_text[self.train_pointer * self.batch_size:
            #                                     (self.train_pointer + 1) * self.batch_size]
            # emotion_raw_data = self.train_emotion_raw_text[self.train_pointer * self.batch_size:
            #                                     (self.train_pointer + 1) * self.batch_size]
            nouns_raw_data = self.train_nouns_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
            verbs_raw_data = self.train_verbs_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
            prons_raw_data = self.train_prons_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
            tags_raw_data = self.train_tag_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
        else:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size: ]
            # act_raw_data = self.train_act_raw_text[self.train_pointer * self.batch_size: ]
            # emotion_raw_data = self.train_emotion_raw_text[self.train_pointer * self.batch_size: ]
            nouns_raw_data = self.train_nouns_raw_text[self.train_pointer * self.batch_size: ]
            verbs_raw_data = self.train_verbs_raw_text[self.train_pointer * self.batch_size: ]
            prons_raw_data = self.train_prons_raw_text[self.train_pointer * self.batch_size: ]
            tags_raw_data = self.train_tag_raw_text[self.train_pointer * self.batch_size: ]

        self.train_pointer += 1

        qa_pairs, verbs_list, prons_list, nouns_list, tags_list = \
            self.dialogues_vpn_into_qas(raw_data, verbs_raw_data, prons_raw_data, nouns_raw_data, tags_raw_data)

        qa_pairs = np.asarray(qa_pairs)

        # qa_pairs, verbs_list, prons_list, nouns_list\
        #     = np.asarray(qa_pairs), np.asarray(verbs_list), np.asarray(prons_list), \
        #                     np.asarray(nouns_list)

        if len(qa_pairs) == 0:
            return np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                    np.asarray([None]), np.asarray([None]), np.asarray([None])
        x_batch = qa_pairs[:, 0]
        y_batch = qa_pairs[:, -1]

        x_v_batch = verbs_list
        x_p_batch = prons_list
        x_n_batch = nouns_list
        y_tag_batch = tags_list

        x_length = [len(item) for item in x_batch]
        x_v_length = [len(item) for item in x_v_batch]
        x_p_length = [len(item) for item in x_p_batch]
        x_n_length = [len(item) for item in x_n_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]


        map_y_batch = []
        for idx, batch in enumerate(y_batch):
            y_tag = y_tag_batch[idx]
            map_item = []
            for jdx, ele in enumerate(batch):
                if y_tag[jdx] == '0':
                    cont = self.nouns_vocab_id.get(self.id_vocab[ele], UNK_ID)
                elif y_tag[jdx] == '1':
                    cont = self.verbs_vocab_id.get(self.id_vocab[ele], UNK_ID)
                elif y_tag[jdx] == '2':
                    cont = self.prons_vocab_id.get(self.id_vocab[ele], UNK_ID)
                else:
                    cont = ele
                map_item.append(cont)
            map_y_batch.append(map_item)


        go_y_batch = self.add_go(y_batch)
        eos_map = self.add_eos_tag(map_y_batch, 3)
        eos_y_tag_batch = self.str_into_int(self.add_eos_tag(y_tag_batch, '3'))

        # y_max_length = np.amax(y_length)

        return x_batch, x_v_batch, x_p_batch, x_n_batch, x_length, x_v_length, x_p_length, \
               x_n_length, go_y_batch, y_length, eos_map, eos_y_tag_batch

    def get_validation(self):
        if self.val_pointer < self.val_batch_num:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
            # act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size:
            #                                     (self.val_pointer + 1) * self.val_batch_size]
            # emotion_raw_data = self.validation_emotion_raw_text[self.val_pointer * self.val_batch_size:
            #                                     (self.val_pointer + 1) * self.val_batch_size]
            nouns_raw_data = self.val_nouns_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
            verbs_raw_data = self.val_verbs_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
            prons_raw_data = self.val_prons_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
            tags_raw_data = self.val_tag_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]

        else:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size: ]
            # act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size: ]
            # emotion_raw_data = self.validation_emotion_raw_text[self.val_pointer * self.val_batch_size: ]
            nouns_raw_data = self.val_nouns_raw_text[self.val_pointer * self.val_batch_size: ]
            verbs_raw_data = self.val_verbs_raw_text[self.val_pointer * self.val_batch_size: ]
            prons_raw_data = self.val_prons_raw_text[self.val_pointer * self.val_batch_size: ]
            tags_raw_data = self.val_tag_raw_text[self.val_pointer * self.val_batch_size: ]

        self.val_pointer += 1

        qa_pairs, verbs_list, prons_list, nouns_list, tags_list = \
            self.dialogues_vpn_into_qas(raw_data, verbs_raw_data, prons_raw_data, nouns_raw_data, tags_raw_data)

        qa_pairs = np.asarray(qa_pairs)

        # qa_pairs, verbs_list, prons_list, nouns_list\
        #     = np.asarray(qa_pairs), np.asarray(verbs_list), np.asarray(prons_list), \
        #                     np.asarray(nouns_list)

        if len(qa_pairs) == 0:
            return np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                    np.asarray([None]), np.asarray([None]), np.asarray([None])
        x_batch = qa_pairs[:, 0]
        y_batch = qa_pairs[:, -1]

        x_v_batch = verbs_list
        x_p_batch = prons_list
        x_n_batch = nouns_list
        y_tag_batch = tags_list

        x_length = [len(item) for item in x_batch]
        x_v_length = [len(item) for item in x_v_batch]
        x_p_length = [len(item) for item in x_p_batch]
        x_n_length = [len(item) for item in x_n_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]


        map_y_batch = []
        for idx, batch in enumerate(y_batch):
            y_tag = y_tag_batch[idx]
            map_item = []
            for jdx, ele in enumerate(batch):
                if y_tag[jdx] == '0':
                    cont = self.nouns_vocab_id.get(self.id_vocab[ele], UNK_ID)
                elif y_tag[jdx] == '1':
                    cont = self.verbs_vocab_id.get(self.id_vocab[ele], UNK_ID)
                elif y_tag[jdx] == '2':
                    cont = self.prons_vocab_id.get(self.id_vocab[ele], UNK_ID)
                else:
                    cont = ele
                map_item.append(cont)
            map_y_batch.append(map_item)


        go_y_batch = self.add_go(y_batch)
        eos_map = self.add_eos_tag(map_y_batch, 3)
        eos_y_tag_batch = self.str_into_int(self.add_eos_tag(y_tag_batch, '3'))

        # y_max_length = np.amax(y_length)

        return x_batch, x_v_batch, x_p_batch, x_n_batch, x_length, x_v_length, x_p_length, \
               x_n_length, go_y_batch, y_length, eos_map, eos_y_tag_batch

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

    def add_eos(self, sentences):
        eos_sentences = []
        for sentence in sentences:
            new_sentence = copy.copy(sentence)
            new_sentence.append(EOS_ID)
            eos_sentences.append(new_sentence)
        return eos_sentences

    def str_into_int(self, sentences):
        eos_sentences = []
        for sentence in sentences:
            new_sentence = [int(item) for item in sentence]
            eos_sentences.append(new_sentence)
        return eos_sentences

    def add_eos_tag(self, sentences, tag):
        eos_sentences = []
        for sentence in sentences:
            new_sentence = copy.copy(sentence)
            new_sentence.append(tag)
            eos_sentences.append(new_sentence)
        return eos_sentences

    def add_go(self, sentences):
        go_sentences = []
        for sentence in sentences:
            new_sentence = copy.copy(sentence)
            new_sentence.insert(0, GO_ID)
            go_sentences.append(new_sentence)
        return go_sentences


    def reset_pointer(self):
        self.train_pointer = 0
        self.val_pointer = 0


class Generator(nn.Module):
    def __init__(self, embeddings):
        super(Generator, self).__init__()
        self.hidden_size = EMBEDDING_SIZE
        self.sen_output_size = VOCAB_SIZE
        self.noun_output_size = 10000
        self.pron_output_size = 82
        self.verb_output_size = 6555
        self.tag_size = 4

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.encoder = nn.GRU(self.hidden_size, self.hidden_size)

        self.tag_decoder = nn.GRU(self.tag_size, self.tag_size)
        self.sen_state_tag_layer = nn.Linear(self.hidden_size, self.tag_size)

        self.decoder_sate_layer = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.docoder = nn.GRU(self.hidden_size, self.hidden_size)
        self.sen_output_layer = nn.Linear(self.hidden_size, self.sen_output_size)
        self.noun_output_layer = nn.Linear(self.hidden_size, self.noun_output_size)
        self.pron_output_layer = nn.Linear(self.hidden_size, self.pron_output_size)
        self.verb_output_layer = nn.Linear(self.hidden_size, self.verb_output_size)

    def forward(self, verb_encoder_input, pron_encoder_input, noun_encoder_input,
                sen_encoder_input, target_word_input, target_tag_input):
        verb_encoder_embed = self.embedding_layer(verb_encoder_input)
        pron_encoder_embed = self.embedding_layer(pron_encoder_input)
        noun_encoder_embed = self.embedding_layer(noun_encoder_input)
        sen_encoder_embed = self.embedding_layer(sen_encoder_input)

        target_embed = self.embedding_layer(target_word_input)

        hidden = None
        for idx, word in enumerate(verb_encoder_embed):
            output, hidden = self.encoder(word.view(1, 1, -1), hidden)
        verb_hidden = hidden


        hidden = None
        for idx, word in enumerate(pron_encoder_embed):
            output, hidden = self.encoder(word.view(1, 1, -1), hidden)
        pron_hidden = hidden

        hidden = None
        for idx, word in enumerate(noun_encoder_embed):
            output, hidden = self.encoder(word.view(1, 1, -1), hidden)
        noun_hidden = hidden

        hidden = None
        for idx, word in enumerate(sen_encoder_embed):
            output, hidden = self.encoder(word.view(1, 1, -1), hidden)
        sen_hidden = hidden

        hidden = self.decoder_sate_layer(
            torch.cat((verb_hidden, pron_hidden, noun_hidden, sen_hidden), dim=-1)
        )

        outputs = []
        for idx, word in enumerate(target_embed):
            output, hidden = self.docoder(word.view(1, 1, -1), hidden)
            index = target_tag_input[idx]
            if index == 0:
                mid_out = self.noun_output_layer(output[0])
            elif index == 1:
                mid_out = self.verb_output_layer(output[0])
            elif index == 2:
                mid_out = self.pron_output_layer(output[0])
            else:
                mid_out = self.sen_output_layer(output[0])
            outputs.append(mid_out)
        return outputs

EPOCH_SIZE = 50

prefix = 'vpn_sample'


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

    if is_toy:
        generator = Generator(data_loader.embedding_matrix)
        # generator.load_state_dict(torch.load(model_prefix%'2'))
    else:
        generator = Generator(data_loader.embedding_matrix).cuda()
        # generator.load_state_dict(torch.load(model_prefix%'7'))
        # torch.backends.cudnn.enabled = False
        # print('Model has been loaded')
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)

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
            x_batch, x_v_batch, x_p_batch, x_n_batch, x_length, x_v_length, x_p_length, \
               x_n_length, go_y_batch, y_length, eos_map, eos_y_tag_batch = data_loader.get_batch_data()

            if x_batch.all() == None:
                continue

            for idx, x_len_ele in enumerate(x_length):

                # if not is_toy:
                #     print('train: %d batch and %d sentence' % (bn, idx))

                loss_mean = 0

                if is_toy:
                    x_batch_, x_v_batch_, x_p_batch_, x_n_batch_, go_y_batch_, eos_map_, eos_y_tag_ = \
                        torch.LongTensor(x_batch[idx]), \
                        torch.LongTensor(x_v_batch[idx]), \
                        torch.LongTensor(x_p_batch[idx]), \
                        torch.LongTensor(x_n_batch[idx]), \
                        torch.LongTensor(go_y_batch[idx]), \
                        torch.LongTensor(eos_map[idx]), \
                        torch.LongTensor(eos_y_tag_batch[idx])
                else:
                    x_batch_, x_v_batch_, x_p_batch_, x_n_batch_, go_y_batch_, eos_map_, eos_y_tag_ = \
                        torch.cuda.LongTensor(x_batch[idx]), \
                        torch.cuda.LongTensor(x_v_batch[idx]), \
                        torch.cuda.LongTensor(x_p_batch[idx]), \
                        torch.cuda.LongTensor(x_n_batch[idx]), \
                        torch.cuda.LongTensor(go_y_batch[idx]), \
                        torch.cuda.LongTensor(eos_map[idx]), \
                        torch.cuda.LongTensor(eos_y_tag_batch[idx])

                step += 1

                outputs = generator(x_v_batch_, x_p_batch_, x_n_batch_, x_batch_, go_y_batch_, eos_y_tag_)

                for jdx, eos_map_y_word in enumerate(eos_map_):
                    print(outputs[jdx].shape)
                    loss_mean += loss_func(outputs[jdx], eos_map_y_word.view(1,))

                generator_optimizer.zero_grad()
                loss_mean.backward()
                generator_optimizer.step()

                losses += loss_mean.item() / y_length[idx]

        generator.eval()

        for bn in range(data_loader.val_batch_num + 1):
            x_batch, x_v_batch, x_p_batch, x_n_batch, x_length, x_v_length, x_p_length, \
               x_n_length, go_y_batch, y_length, eos_map, eos_y_tag_batch = data_loader.get_validation()
            if x_batch.all() == None:
                continue

            for idx, x_len_ele in enumerate(x_length):

                # if not is_toy:
                #     print('validation: %d batch and %d sentence' % (bn, idx))

                val_loss_mean = 0

                if is_toy:
                    x_batch_, x_v_batch_, x_p_batch_, x_n_batch_, go_y_batch_, eos_map_, eos_y_tag_ = \
                        torch.LongTensor(x_batch[idx]), \
                        torch.LongTensor(x_v_batch[idx]), \
                        torch.LongTensor(x_p_batch[idx]), \
                        torch.LongTensor(x_n_batch[idx]), \
                        torch.LongTensor(go_y_batch[idx]), \
                        torch.LongTensor(eos_map[idx]), \
                        torch.LongTensor(eos_y_tag_batch[idx])
                else:
                    x_batch_, x_v_batch_, x_p_batch_, x_n_batch_, go_y_batch_, eos_map_, eos_y_tag_ = \
                        torch.cuda.LongTensor(x_batch[idx]), \
                        torch.cuda.LongTensor(x_v_batch[idx]), \
                        torch.cuda.LongTensor(x_p_batch[idx]), \
                        torch.cuda.LongTensor(x_n_batch[idx]), \
                        torch.cuda.LongTensor(go_y_batch[idx]), \
                        torch.cuda.LongTensor(eos_map[idx]), \
                        torch.cuda.LongTensor(eos_y_tag_batch[idx])

                step += 1

                outputs = generator(x_v_batch_, x_p_batch_, x_n_batch_, x_batch_, go_y_batch_, eos_y_tag_)

                for jdx, eos_map_y_word in enumerate(eos_map_):
                    val_loss_mean += loss_func(outputs[jdx], eos_map_y_word.view(1,))

                generator_optimizer.zero_grad()

                val_step += 1
                val_losses += val_loss_mean.item() / y_length[idx]


        print("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}".format(epoch + 1,
                                    EPOCH_SIZE, losses / step, val_losses / val_step))
        log.write("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}\n".format(epoch + 1,
                                    EPOCH_SIZE, losses / step, val_losses / val_step))

        torch.save(generator.state_dict(), checkpoint_dir + 'generator_para_' + str(epoch + 1) + '.pkl')
        print('Model Trained and Saved in epoch ', epoch + 1)

        data_loader.reset_pointer()

    log.close()

def main_test(is_toy=False):
    data_loader = DataLoader(is_toy)

    if not is_toy:
        import sys
        reload(sys)
        sys.setdefaultencoding('utf8')

    res_file = 'models_' + prefix + '/results.txt'
    res = codecs.open(res_file, 'w')

    reply_file = 'models_' + prefix + '/vpn_reply.txt'
    reply_f = codecs.open(reply_file, 'w')

    ans_file = 'models_' + prefix + '/vpn_answer.txt'
    ans_f = codecs.open(ans_file, 'w')

    load_index = '4'

    # test
    print('test')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    else:
        data_loader.load_embedding()
    print('load the embedding matrix')

    checkpoint_file = 'models_' + prefix + '/generator_para_%s.pkl'
    if is_toy:
        generator = Generator(data_loader.embedding_matrix)
    else:
        generator = Generator(data_loader.embedding_matrix).cuda()
    generator.load_state_dict(torch.load(checkpoint_file % load_index))
    print('Model has been restored')

    all_test_reply = []

    for bn in range(data_loader.test_batch_num + 1):
        x_batch, x_v_batch, x_p_batch, x_n_batch, x_length, x_v_length, x_p_length, \
               x_n_length, go_y_batch, y_length, eos_map, eos_y_tag_batch = data_loader.get_batch_test()
        if x_batch.all() == None:
            continue

        all_reply = []

        print('=========================================================')
        res.write('=========================================================\n')

        for idx, x_len_ele in enumerate(x_length):

            print('test: %d batch and %d sentence' % (bn, idx))

            if is_toy:
                x_batch_, x_v_batch_, x_p_batch_, x_n_batch_, go_y_batch_, eos_map_, eos_y_tag_ = \
                    torch.LongTensor(x_batch[idx]), \
                    torch.LongTensor(x_v_batch[idx]), \
                    torch.LongTensor(x_p_batch[idx]), \
                    torch.LongTensor(x_n_batch[idx]), \
                    torch.LongTensor(go_y_batch[idx]), \
                    torch.LongTensor(eos_map[idx]), \
                    torch.LongTensor(eos_y_tag_batch[idx])
            else:
                x_batch_, x_v_batch_, x_p_batch_, x_n_batch_, go_y_batch_, eos_map_, eos_y_tag_ = \
                    torch.cuda.LongTensor(x_batch[idx]), \
                    torch.cuda.LongTensor(x_v_batch[idx]), \
                    torch.cuda.LongTensor(x_p_batch[idx]), \
                    torch.cuda.LongTensor(x_n_batch[idx]), \
                    torch.cuda.LongTensor(go_y_batch[idx]), \
                    torch.cuda.LongTensor(eos_map[idx]), \
                    torch.cuda.LongTensor(eos_y_tag_batch[idx])

            outputs = generator(x_v_batch_, x_p_batch_, x_n_batch_, x_batch_, go_y_batch_, eos_y_tag_)

            reply = []
            for jdx, ele in enumerate(outputs):
                id_word = torch.argmax(ele).item()
                if id_word != PAD_ID and id_word != EOS_ID:
                    reply.append(data_loader.id_vocab[id_word])

            all_reply.append(reply)

            question = [data_loader.id_vocab[id_word.item()] for id_word in x_batch_]
            answer = [data_loader.id_vocab[id_word.item()] for id_word in go_y_batch_ if id_word != GO_ID]

            print('Question:')
            res.write('Question:\n')
            print(question)
            res.write(' '.join(question) + '\n')
            print('Answer:')
            res.write('Answer:\n')
            print(answer)
            res.write(' '.join(answer) + '\n')
            ans_f.write(' '.join(answer) + '\n')
            print('Generation:')
            res.write('Generation:\n')
            print(reply)
            res.write(' '.join(reply) + '\n')
            reply_f.write(' '.join(reply) + '\n')

            print('---------------------------------------------')
            res.write('---------------------------------------------\n')

        all_test_reply.extend(all_reply)


    res.close()
    reply_f.close()
    ans_f.close()

if __name__ == '__main__':
    main_train(True)

    # main_test(True)

    # main_train()

    # main_test()

    # d = DataLoader()
    # for _ in range(d.batch_num + 1):
    #     x_batch, x_v_batch, x_p_batch, x_n_batch, x_length, x_v_length, x_p_length, \
    #            x_n_length, go_y_batch, y_length, eos_map, eos_y_tag_batch = d.get_validation()
    #     print('x_batch', len(x_batch))
    #     print('x_v_batch', len(x_v_batch))
    #     print('x_p_batch', len(x_p_batch))
    #     print('x_n_batch', len(x_n_batch))
    #     print('x_length', len(x_length))
    #     print('x_v_length', len(x_v_length))
    #     print('x_p_length', len(x_p_length))
    #     print('x_n_length', len(x_n_length))
    #     print('go_y_batch', len(go_y_batch))
    #     print('y_length', len(y_length))
    #     print('eos_map', len(eos_map))
    #     print('eos_y_tag_batch', len(eos_y_tag_batch))
