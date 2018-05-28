# coding=utf-8
__author__ = 'yhd'

import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
            self.source_train_act = 'data_root/act_train.txt'
            self.source_val_act = 'data_root/act_val.txt'
            self.source_test_act = 'data_root/act_test.txt'
            self.source_train_emotion = 'data_root/emotion_train.txt'
            self.source_val_emotion = 'data_root/emotion_validation.txt'
            self.source_test_emotion = 'data_root/emotion_test.txt'
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
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None])
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
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None])
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

    def batch_emotions_into_items(self, dialogues, dia_acts):
        sen_list = []
        act_list = []
        for idx, ele in enumerate(dia_acts):
            sen_acts = ele.split()
            sentences = dialogues[idx].split('__eou__')[:-1]
            for jdx, item in enumerate(sen_acts):
                sen_list.append(self.sentence_to_token_ids(sentences[jdx]))
                act_list.append(int(item) -  1)
        return sen_list, act_list

    def get_batch_emotion_data(self):
        if self.train_pointer < self.batch_num:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
            act_raw_data = self.train_act_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
        else:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size: ]
            act_raw_data = self.train_act_raw_text[self.train_pointer * self.batch_size: ]

        self.train_pointer += 1

        sen_batch, class_batch = self.batch_emotions_into_items(raw_data, act_raw_data)
        sen_batch, class_batch = np.asarray(sen_batch), np.asarray(class_batch)
        if sen_batch.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None])

        sen_length = [len(item) for item in sen_batch]

        return np.asarray(self.pad_sentence(sen_batch, np.amax(sen_length))), np.asarray(sen_length), \
                np.asarray(class_batch)

    def get_emotion_validation(self):
        if self.val_pointer < self.val_batch_num:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
            act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
        else:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size: ]
            act_raw_data = self.validation_act_raw_text[self.val_pointer * self.val_batch_size: ]

        self.val_pointer += 1

        sen_batch, class_batch = self.batch_emotions_into_items(raw_data, act_raw_data)
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

class Generator(nn.Module):
    def __init__(self, embeddings):
        super(Generator, self).__init__()
        self.hidden_size = EMBEDDING_SIZE
        self.output_size = VOCAB_SIZE

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.encoder = nn.GRU(self.hidden_size, self.hidden_size)

        self.docoder = nn.GRU(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_input, target_input):
        encoder_embed = self.embedding_layer(encoder_input)
        target_embed = self.embedding_layer(target_input)
        hidden = None
        for idx, word in enumerate(encoder_embed):
            output, hidden = self.encoder(word.view(1, 1, -1), hidden)

        # emb = self.embedding_layer(torch.cuda.LongTensor([GO_ID]))
        outputs = []
        for idx, word in enumerate(target_embed):
            output, hidden = self.docoder(word.view(1, 1, -1), hidden)
            outputs.append(self.output_layer(output[0]))

            # output, hidden = self.docoder(emb.view(1, 1, -1), hidden)
            # output_vocab = output[0]
            # output_id = torch.argmax(output_vocab)
            # emb = self.embedding_layer(output_id)
            # if output_id.item() == EOS_ID:
            #     break
            # outputs.append(output_vocab)
        return outputs

EPOCH_SIZE = 50

prefix = 'seq_torch_seq'

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
        generator.load_state_dict(torch.load(model_prefix%'3'))
        print('Model 3 has been loaded')
    # generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.00001)


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
            pad_x_batch, x_length, eos_pad_y_batch,  go_pad_y_batch, y_length, \
                            x_act_batch, y_act_batch, x_emotion_batch, \
                            y_emotion_batch = data_loader.get_batch_data()

            if pad_x_batch.all() == None:
                continue

            for idx, x_len_ele in enumerate(x_length):

                loss_mean = 0

                if is_toy:
                    pad_x, eos_pad_y, go_pad_y = \
                        torch.LongTensor(pad_x_batch[idx]), \
                        torch.LongTensor(eos_pad_y_batch[idx]), \
                        torch.LongTensor(go_pad_y_batch[idx])
                else:
                    pad_x, eos_pad_y, go_pad_y = \
                        torch.cuda.LongTensor(pad_x_batch[idx]), \
                        torch.cuda.LongTensor(eos_pad_y_batch[idx]), \
                        torch.cuda.LongTensor(go_pad_y_batch[idx])

                step += 1

                outputs = generator(pad_x, go_pad_y)

                for jdx, eos_y_word in enumerate(eos_pad_y):
                    loss_mean += loss_func(outputs[jdx], eos_pad_y[jdx].view(1,))

                generator_optimizer.zero_grad()
                loss_mean.backward()
                generator_optimizer.step()

                losses += loss_mean.item() / len(outputs)

        generator.eval()

        for _ in range(data_loader.val_batch_num + 1):
            pad_x_batch, x_length, eos_pad_y_batch,  go_pad_y_batch, y_length, \
                            x_act_batch, y_act_batch, x_emotion_batch, \
                            y_emotion_batch = data_loader.get_validation()
            if pad_x_batch.all() == None:
                continue

            for idx, x_len_ele in enumerate(x_length):

                val_loss_mean = 0

                if is_toy:
                    pad_x, eos_pad_y, go_pad_y = \
                        torch.LongTensor(pad_x_batch[idx]), \
                        torch.LongTensor(eos_pad_y_batch[idx]), \
                        torch.LongTensor(go_pad_y_batch[idx])
                else:
                    pad_x, eos_pad_y, go_pad_y = \
                        torch.cuda.LongTensor(pad_x_batch[idx]), \
                        torch.cuda.LongTensor(eos_pad_y_batch[idx]), \
                        torch.cuda.LongTensor(go_pad_y_batch[idx])

                outputs = generator(pad_x, go_pad_y)

                for jdx, eos_y_word in enumerate(eos_pad_y):
                    val_loss_mean += loss_func(outputs[jdx], eos_pad_y[jdx].view(1,))

                generator_optimizer.zero_grad()

                val_step += 1
                val_losses += val_loss_mean.item() / len(outputs)


        print("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}".format(epoch + 1,
                                    EPOCH_SIZE, losses / step, val_losses / val_step))
        log.write("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}\n".format(epoch + 1,
                                    EPOCH_SIZE, losses / step, val_losses / val_step))

        torch.save(generator.state_dict(), checkpoint_dir + 'generator_sgd_para_' + str(epoch + 1) + '.pkl')
        print('Model Trained and Saved in epoch ', epoch + 1)

        data_loader.reset_pointer()

    log.close()


def main_test(is_toy=False):
    data_loader = DataLoader(is_toy)

    if not is_toy:
        import sys
        reload(sys)
        sys.setdefaultencoding('utf8')

    res_file = 'models_' + prefix + '/seq_torch_seq_results.txt'
    res = codecs.open(res_file, 'w')

    reply_file = 'models_' + prefix + '/seq_torch_seq_reply.txt'
    reply_f = codecs.open(reply_file, 'w')

    ans_file = 'models_' + prefix + '/seq_torch_seq_answer.txt'
    ans_f = codecs.open(ans_file, 'w')

    load_index = '3'

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

    for _ in range(data_loader.test_batch_num + 1):
        pad_x_batch, x_length, \
            eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_batch_test()
        if pad_x_batch.all() == None:
            continue

        all_reply = []

        for idx, x_len_ele in enumerate(x_length):

            if is_toy:
                pad_x, eos_pad_y, go_pad_y = \
                        torch.LongTensor(pad_x_batch[idx]), \
                        torch.LongTensor(eos_pad_y_batch[idx]), \
                        torch.LongTensor(go_pad_y_batch[idx])
            else:
                pad_x, eos_pad_y, go_pad_y = \
                        torch.cuda.LongTensor(pad_x_batch[idx]), \
                        torch.cuda.LongTensor(eos_pad_y_batch[idx]), \
                        torch.cuda.LongTensor(go_pad_y_batch[idx])

            outputs = generator(pad_x, go_pad_y)

            reply = []
            for jdx, ele in enumerate(outputs):
                id_word = torch.argmax(ele).item()
                if id_word != PAD_ID and id_word != EOS_ID:
                    reply.append(data_loader.id_vocab[id_word])

            all_reply.append(reply)

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

if __name__ == '__main__':
    # main_train(True)

    # main_test(True)

    main_train()

    # main_test()
