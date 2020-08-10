# coding=utf-8
__author__ = 'yhd'

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import platform

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
        self.embedding_matrix = torch.FloatTensor(np.asarray(lookup_table))

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

        return np.asarray(sen_batch), np.asarray(sen_length), \
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

        return np.asarray(sen_batch), np.asarray(sen_length), \
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

        return np.asarray(sen_batch), np.asarray(sen_length), \
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


class DAN_ACT(nn.Module):

    def __init__(self, embeddings):
        super(DAN_ACT, self).__init__()
        self.embedding_size = EMBEDDING_SIZE
        layer1_units = 100
        self.act_class_num = 4
        self.drop_out = 0.2

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.hidden_layers = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size),
                                    nn.Dropout(self.drop_out))
        self.output_layer = nn.Sequential(nn.Linear(self.embedding_size, self.act_class_num),
                                    nn.Dropout(self.drop_out))

        self.inter_h_emb = nn.Bilinear(self.embedding_size, self.embedding_size, 1)

    def forward(self, x_batch):
        embed = self.embedding_layer(x_batch)
        dense_inputs = torch.mean(embed, dim=1)
        mid_rep = self.hidden_layers(dense_inputs)

        reward = []
        for embed_word in embed[0]:
            a_i = self.inter_h_emb(embed_word.view(1, -1), mid_rep)
            reward.append(a_i)

        reward = torch.stack(reward).view(1, -1)
        reward = F.softmax(reward, dim=1)
        addition = torch.mean(reward.view(1, -1, 1) * embed, dim=1)
        output = self.output_layer(mid_rep + addition)

        return output, reward

EPOCH_SIZE = 50

prefix = 'adan_act_softmax'


def main_train(is_toy=False):

    # data
    data_loader = DataLoader(is_toy)
    print('train')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    else:
        data_loader.load_embedding()
    print('load the embedding matrix')

    checkpoint_dir = 'models_' + prefix + '/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    log_file = checkpoint_dir + 'log.txt'
    log = codecs.open(log_file, 'a')

    model_prefix = checkpoint_dir + 'adan_para_%s.pkl'

    # model
    if is_toy:
        model = DAN_ACT(data_loader.embedding_matrix)
    else:
        model = DAN_ACT(data_loader.embedding_matrix).cuda()
        # model.load_state_dict(torch.load(model_prefix%'9'))
        # print('Model 9 has been loaded')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
    loss_func = nn.CrossEntropyLoss()


    # train
    for epoch in range(EPOCH_SIZE):
        losses = 0
        step = 0
        val_losses = 0
        val_step = 0
        accs = 0

        model.train()

        for _ in range(data_loader.batch_num + 1):
            pad_sen_batch, sen_length, act_label = data_loader.get_batch_data()

            if pad_sen_batch.all() == None:
                continue

            for idx, length in enumerate(sen_length):
                sen_batch = [pad_sen_batch[idx]]
                act_label_ = [act_label[idx]]

                step += 1

                if is_toy:
                    sen_batch, act_label_ = torch.LongTensor(sen_batch), torch.LongTensor(act_label_)
                else:
                    sen_batch, act_label_ = torch.LongTensor(sen_batch).cuda(), \
                                             torch.LongTensor(act_label_).cuda()

                out, rew = model(sen_batch)

                loss_mean = loss_func(out, act_label_)

                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()

                losses += loss_mean

        model.eval()

        for _ in range(data_loader.val_batch_num + 1):
            pad_sen_batch, sen_length, act_label = data_loader.get_validation()
            if pad_sen_batch.all() == None:
                continue

            for idx, length in enumerate(sen_length):
                sen_batch = [pad_sen_batch[idx]]
                act_label_ = [act_label[idx]]

                if is_toy:
                    sen_batch, act_label_ = torch.LongTensor(sen_batch), torch.LongTensor(act_label_)
                else:
                    sen_batch, act_label_ = torch.LongTensor(sen_batch).cuda(), \
                                             torch.LongTensor(act_label_).cuda()

                out, rew = model(sen_batch)

                if is_toy:
                    acc = torch.mean(torch.eq(torch.argmax(out, dim=1), act_label_).float())
                else:
                    acc = torch.mean(torch.eq(torch.argmax(out, dim=1), act_label_).float()).cuda()

                val_loss_mean = loss_func(out, act_label_)

                optimizer.zero_grad()

                val_step += 1
                val_losses += val_loss_mean
                accs += acc


        print("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g} Valid Acc {:g}".format(epoch + 1,
                                    EPOCH_SIZE, losses / step, val_losses / val_step, accs / val_step))
        log.write("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g} Valid Acc {:g}\n".format(epoch + 1,
                                    EPOCH_SIZE, losses / step, val_losses / val_step, accs / val_step))

        torch.save(model.state_dict(), checkpoint_dir + 'adan_para_' + str(epoch + 1) + '.pkl')
        print('Model Trained and Saved in epoch ', epoch + 1)

        data_loader.reset_pointer()

    log.close()


def main_test(is_toy=False):
    data_loader = DataLoader(is_toy)

    if not is_toy:
        import sys
        reload(sys)
        sys.setdefaultencoding('utf8')

    res_file = 'models_' + prefix + '/adan_act_softmax_results.txt'
    res = codecs.open(res_file, 'w')

    load_index = '6'

    # test
    print('test')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    else:
        data_loader.load_embedding()
    print('load the embedding matrix')

    checkpoint_file = 'models_' + prefix + '/generator_sgd_para_%s.pkl'
    if is_toy:
        model = DAN_ACT(data_loader.embedding_matrix)
    else:
        model = DAN_ACT(data_loader.embedding_matrix).cuda()
    model.load_state_dict(torch.load(checkpoint_file % load_index))
    print('Model has been restored')

    for _ in range(data_loader.test_batch_num + 1):
        pad_sen_batch, sen_length, act_label = data_loader.get_batch_data()

        if pad_sen_batch.all() == None:
            continue

        print('=========================================================')
        res.write('=========================================================\n')

        for idx, length in enumerate(sen_length):
            sen_batch = pad_sen_batch[idx]
            act_label_ = act_label[idx]

            if is_toy:
                sen_batch, act_label_ = torch.LongTensor(sen_batch), torch.LongTensor(act_label_)
            else:
                sen_batch, act_label_ = torch.LongTensor(sen_batch).cuda(), \
                                         torch.LongTensor(act_label_).cuda()

            outputs, rewards = model(sen_batch)

            sentence = [data_loader.id_vocab[id_word] for id_word in sen_batch]

            print('Sentence:')
            res.write('Sentence:\n')
            print(' '.join(sentence))
            res.write(' '.join(sentence) + '\n')
            print('Act:')
            res.write('Act:\n')
            print(act_id_act(act_label_))
            res.write(act_id_act(act_label_) + '\n')
            print('Reward:')
            res.write('Reward:\n')
            print(' '.join(rewards))
            res.write(' '.join(rewards) + '\n')

            print('---------------------------------------------')
            res.write('---------------------------------------------\n')

    res.close()

def act_id_act(act_id):
    if act_id == 0:
        res = 'inform'
    elif act_id == 1:
        res = 'question'
    elif act_id == 2:
        res = 'directive'
    else:
        res = 'commissive'
    return res

if __name__ == '__main__':
    # main_train(True)

    main_train()

    # main_test()
