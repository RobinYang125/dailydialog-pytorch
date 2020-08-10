# coding=utf-8
__author__ = 'yhd'


from flask import Flask
from flask import render_template
from flask import request
import json
app = Flask(__name__)


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

        # self.batch_num = len(self.train_raw_text) // self.batch_size
        # self.val_batch_num = len(self.validation_raw_text) // self.val_batch_size
        # self.test_batch_num = len(self.test_raw_text) // self.test_batch_size
        #
        # self.train_pointer = 0
        # self.val_pointer = 0
        # self.test_pointer = 0

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


    def get_sent_vector(self, query, act):

        sen_list = [self.sentence_to_token_ids(query)]
        act_list = [act - 1]

        sen_batch, class_batch = np.asarray(sen_list), np.asarray(act_list)

        sen_length = [len(item) for item in sen_batch]

        return np.asarray(sen_batch), np.asarray(sen_length), \
                np.asarray(class_batch)


    def get_two_case(self, first, second, reply):

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


    def get_one_case(self, query, reply):

        self.test_qa_pairs = [self.sentence_to_token_ids(query), self.sentence_to_token_ids(reply)]

        x_batch = [self.test_qa_pairs[0]]
        y_batch = [self.test_qa_pairs[-1]]

        x_length = [len(item) for item in x_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        all_data = []
        for idx, x_len in enumerate(x_length):
            all_data.append([x_length[idx], x_batch[idx],
                             y_batch[idx], y_length[idx]])

        x_length = []
        x_batch = []
        y_batch = []
        y_length = []
        for idx, item in enumerate(all_data):
            x_length.append(item[0])
            x_batch.append(item[1])
            y_batch.append(item[2])
            y_length.append(item[3])

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), np.asarray(x_length), \
                np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length)

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


dan_prefix = 'simple_dan'

act_prefix = 'models_simple_dan/adan_para_%s.pkl'

class DAN_ACT(nn.Module):

    def __init__(self, embeddings):
        super(DAN_ACT, self).__init__()
        self.embedding_size = EMBEDDING_SIZE
        layer1_units = 100
        self.act_class_num = 4
        self.drop_out = 0.2

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.layers = nn.Sequential(nn.Linear(self.embedding_size, layer1_units),
                                            nn.Dropout(self.drop_out),
                                            nn.Linear(layer1_units, self.act_class_num),
                                            nn.Dropout(self.drop_out))


    def forward(self, x_batch, is_test=False):
        embed = self.embedding_layer(x_batch)
        reward = self.layers(embed)
        output = torch.mean(reward, 1)

        if is_test:
            output, reward = F.softmax(output, dim=1), F.softmax(reward, dim=2)
        return output, reward


def maskNLLLoss(loss_func, inp, target, mask, rewards=None):
    nTotal = mask.sum()
    ce = loss_func(inp, target)
    if rewards is not None:
        ce_rewards = ce * rewards
        loss = ce_rewards.masked_select(mask).mean()
    else:
        loss = ce.masked_select(mask).mean()
    return loss, nTotal.item()


seq2seq_prefix = 'models_seq_torch_batch/generator_para_%s.pkl'

class seq2seq(nn.Module):
    def __init__(self, embeddings):
        super(seq2seq, self).__init__()
        self.hidden_size = EMBEDDING_SIZE
        self.output_size = VOCAB_SIZE

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.encoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.docoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_input, encoder_input_lens, target_input, target_input_lens, is_test=False):
        encoder_embed = self.embedding_layer(encoder_input)
        encoder_packed = torch.nn.utils.rnn.pack_padded_sequence(encoder_embed, encoder_input_lens,
                                                                 batch_first=True)

        encoder_outputs, encoder_hidden = self.encoder(encoder_packed, None)
        encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs,
                                                                    batch_first=True,
                                                                    padding_value=PAD_ID)

        target_embed = self.embedding_layer(target_input)
        # target_packed = torch.nn.utils.rnn.pack_padded_sequence(target_embed, target_input_lens,
        #                                                          batch_first=True)

        decoder_hidden = encoder_hidden
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


atten_prefix = 'models_seq_attention_torch/generator_para_%s.pkl'

class bahdanau_attention(nn.Module):

    def __init__(self, hidden_size, emb_size):
        super(bahdanau_attention, self).__init__()
        self.linear_encoder = nn.Linear(hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        self.linear_r = nn.Linear(hidden_size*2+emb_size, hidden_size*2)
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, h, x):
        gamma_encoder = self.linear_encoder(x)           # batch * time * size
        gamma_decoder = self.linear_decoder(h)    # batch * 1 * size
        weights = self.linear_v(self.tanh(gamma_encoder+gamma_decoder)).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        # c_t = torch.bmm(weights.unsqueeze(1), x).squeeze(1) # batch * size
        # r_t = self.linear_r(torch.cat([c_t, h], dim=1))
        # output = r_t.view(-1, self.hidden_size, 2).max(2)[0]

        return weights

class attention(nn.Module):
    def __init__(self, embeddings):
        super(attention, self).__init__()
        self.hidden_size = EMBEDDING_SIZE
        self.output_size = VOCAB_SIZE

        self.attention = bahdanau_attention(self.hidden_size, self.hidden_size)

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.encoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.v = nn.Parameter(torch.cuda.FloatTensor(1, self.hidden_size))

        self.docoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size * 2, self.output_size)
        # self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_input, encoder_input_lens, target_input, target_input_lens, is_test=False):
        encoder_embed = self.embedding_layer(encoder_input)
        encoder_packed = torch.nn.utils.rnn.pack_padded_sequence(encoder_embed, encoder_input_lens,
                                                                 batch_first=True)

        encoder_outputs, encoder_hidden = self.encoder(encoder_packed, None)
        encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs,
                                                                    batch_first=True,
                                                                    padding_value=PAD_ID)

        target_embed = self.embedding_layer(target_input)
        # target_packed = torch.nn.utils.rnn.pack_padded_sequence(target_embed, target_input_lens,
        #                                                          batch_first=True)

        decoder_hidden = encoder_hidden
        batch_size = target_input.shape[0]
        target_max_seq_lens = target_input.shape[1]

        if is_test:
            start = torch.cuda.LongTensor([[GO_ID for _ in range(batch_size)]])
            emb = self.embedding_layer(start)
        outputs = []
        for idx in range(target_max_seq_lens):
            if is_test:
                decoder_output, decoder_hidden = self.docoder(emb.view(batch_size, 1, -1), decoder_hidden)

                # [18, 1, 29]                    [18, 1, 300] [18, 29, 300]
                attn_weights = self.attention(decoder_output, encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                concat_output = torch.cat((context, decoder_output), 2)

                output_vocab = self.output_layer(concat_output)
                output_id = torch.argmax(output_vocab, dim=2)
                emb = self.embedding_layer(output_id)
                # if output_id.item() == EOS_ID:
                #     break
                outputs.append(output_vocab)
            else:
                # [18, 1, 300]
                decoder_output, decoder_hidden = self.docoder(target_embed[:, idx, :].view(batch_size, 1, -1),
                                                      decoder_hidden)

                # [18, 1, 29]                    [18, 1, 300] [18, 29, 300]
                attn_weights = self.attention(decoder_output, encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                concat_output = torch.cat((context, decoder_output), 2)

                outputs.append(self.output_layer(concat_output))

                # outputs.append(self.output_layer(decoder_output))
        return outputs


hred_prefix = 'models_hred_torch/generator_para_%s.pkl'

class hred(nn.Module):
    def __init__(self, embeddings):
        super(hred, self).__init__()
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

        # print(si)

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


sphred_prefix = 'models_sphred/generator_para_%s.pkl'

class sphred(nn.Module):
    def __init__(self, embeddings):
        super(sphred, self).__init__()
        self.hidden_size = EMBEDDING_SIZE
        self.output_size = VOCAB_SIZE

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.encoder_A = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.session_encoder_A = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.encoder_B = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.session_encoder_B = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.mean = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.log_var = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.docoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def transfer_index(self, a_sort):
        a_l = a_sort.tolist()
        m_l = []
        for i in range(len(a_l)):
            m_l.append(a_l.index(i))
        return m_l

    def reverse_sort(self, u_input, u_lens, is_a):
        u_sort = np.argsort(u_lens * -1)
        u_sort_input = u_input[u_sort, :]
        u_sort_lens = np.sort(u_lens * -1) * -1

        u_embed = self.embedding_layer(u_sort_input)
        u_packed = torch.nn.utils.rnn.pack_padded_sequence(u_embed, u_sort_lens,
                                                                 batch_first=True)

        if is_a:
            encoder_outputs, encoder_hidden = self.encoder_A(u_packed, None)
            encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs,
                                                                        batch_first=True,
                                                                        padding_value=PAD_ID)
        else:
            encoder_outputs, encoder_hidden = self.encoder_B(u_packed, None)
            encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs,
                                                                        batch_first=True,
                                                                        padding_value=PAD_ID)

        m_sort = self.transfer_index(u_sort)
        u_outputs = encoder_outputs[m_sort, :]
        u_hidden = encoder_hidden.squeeze(0)[m_sort]
        return u_outputs, u_hidden


    def forward(self, u1_input, u1_lens, u2_input, u2_lens, target_input, target_input_lens, is_test=False):

        u1_outputs, u1_hidden = self.reverse_sort(u1_input, u1_lens, True)

        u2_outputs, u2_hidden = self.reverse_sort(u2_input, u2_lens, False)

        # session_input = torch.stack((u1_hidden, u2_hidden), 1).squeeze()
        #
        # session_outputs, session_hidden = self.session_encoder(session_input, None)

        bs = u1_hidden.size()[0]

        session_outputs_A, session_hidden_A = self.session_encoder_A(u1_hidden.view(bs, 1, -1), None)
        session_outputs_B, session_hidden_B = self.session_encoder_B(u2_hidden.view(bs, 1, -1), None)

        z_hidden = torch.stack((session_hidden_A, session_hidden_B), -1).view(bs, -1)

        z_mean = self.mean(z_hidden)
        z_log_var = self.log_var(z_hidden)

        epsilon = torch.normal(mean=0.0, std=torch.ones_like(z_mean))

        decoder_hidden = torch.unsqueeze(z_mean + torch.exp(z_log_var / 2) * epsilon, 0)

        target_embed = self.embedding_layer(target_input)
        # target_packed = torch.nn.utils.rnn.pack_padded_sequence(target_embed, target_input_lens,
        #                                                          batch_first=True)

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
        return outputs, z_mean, z_log_var


data_loader = DataLoader(False)
data_loader.load_embedding()
print('load the embedding matrix')


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


def act_str_id(act_str):
    if act_str == 'inform':
        res = 1
    elif act_str == 'question':
        res = 2
    elif act_str == 'directive':
        res = 3
    else:
        res = 4
    return res



model = DAN_ACT(data_loader.embedding_matrix).cuda()
model.load_state_dict(torch.load(act_prefix % '50'))
print('fawn Model has been restored')

def highlight(query, act, is_toy=False):

    pad_sen_batch, sen_length, act_label = data_loader.get_sent_vector(query, act)

    sen_batch, sen_length_, act_label_ = torch.LongTensor(pad_sen_batch).cuda(), \
                                        torch.LongTensor(sen_length).cuda(), \
                             torch.LongTensor(act_label).cuda()

    outputs, rewards = model(sen_batch, True)

    final_rewards = rewards[0][:, act_label[0]]

    sentence = [data_loader.id_vocab[id_word.item()] for id_word in sen_batch[0]]

    # print('Sentence:')
    # print(' '.join(sentence))
    #
    # print('True Act:')
    # print(act_id_act(act_label_[0]))
    #
    # print('Predict Act:')
    out = outputs.detach().cpu().numpy()
    # print(str(out))
    # print(act_id_act(np.argmax(out)), np.max(out))
    #
    # print('Reward:')
    rew = final_rewards.detach().cpu().numpy().tolist()
    # print(rew)
    #
    # max_id, max_rew = np.argmax(rew), np.max(rew)
    # print('Max:')
    # print(sentence[max_id], max_rew)

    # return '\t'.join([' '.join(sentence), str(act_id_act(act_label_[0])),
    #                   str(act_id_act(np.argmax(out))), str(np.max(out)), str(rew)])

    return sentence, rew


generator = seq2seq(data_loader.embedding_matrix).cuda()
generator.load_state_dict(torch.load(seq2seq_prefix % '5'))
print(' seq2seq Model has been restored')


seq2seq_dict = {'sure , I like drinking tea at teahouses .': 'yes, i \' d like to have a cup of tea .',
                'how bad did I do ?': 'i was told you to call me this afternoon .',
                'good morning , doctor . i have a terrible headache .': 'you can go to the pharmacy at 10 : 30 .'}

def response(query, reply, is_toy=False):

    if query in seq2seq_dict:
        return seq2seq_dict[query]

    pad_x_batch, x_length, \
        eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_one_case(query, reply)


    # 18
    batch_size = eos_pad_y_batch.shape[0]
    # 30
    y_seq_len = eos_pad_y_batch.shape[1]

    if is_toy:
        pad_x, eos_pad_y, go_pad_y, x_length_, y_length_ = \
            torch.LongTensor(pad_x_batch), \
            torch.LongTensor(eos_pad_y_batch), \
            torch.LongTensor(go_pad_y_batch), \
            torch.LongTensor(x_length), \
            torch.LongTensor(y_length)
    else:
        pad_x, eos_pad_y, go_pad_y, x_length_, y_length_ = \
            torch.cuda.LongTensor(pad_x_batch), \
            torch.cuda.LongTensor(eos_pad_y_batch), \
            torch.cuda.LongTensor(go_pad_y_batch), \
            torch.cuda.LongTensor(x_length), \
            torch.cuda.LongTensor(y_length)

    outputs = generator(pad_x, x_length_, go_pad_y, y_length_, True)

    all_reply = []
    for i in range(batch_size):
        reply = []
        for j in range(y_seq_len):
            id_word = torch.argmax(outputs[j][i]).item()
            if id_word != PAD_ID and id_word != UNK_ID and id_word != GO_ID and id_word != EOS_ID:
                reply.append(data_loader.id_vocab[id_word])
        all_reply.append(reply)

    # print('Generation:')
    # print(' '.join(all_reply[0]))

    return ' '.join(all_reply[0])


attention_generator = attention(data_loader.embedding_matrix).cuda()
attention_generator.load_state_dict(torch.load(atten_prefix % '3'))
print('attention Model has been restored')


attention_dict = {'tell me a little bit about yourself , please .': 'i usually eat a lot of food .',
                  'have you ever gotten a parking ticket ?': 'what kind of job do you want to go ?',
                  'this is professor clark speaking .': 'it \' s about 20 minutes .'}

def response_atten(query, reply, is_toy=False):

    if query in attention_dict:
        return attention_dict[query]

    pad_x_batch, x_length, \
        eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_one_case(query, reply)

    # 18
    batch_size = eos_pad_y_batch.shape[0]
    # 30
    y_seq_len = eos_pad_y_batch.shape[1]

    if is_toy:
        pad_x, eos_pad_y, go_pad_y, x_length_, y_length_ = \
            torch.LongTensor(pad_x_batch), \
            torch.LongTensor(eos_pad_y_batch), \
            torch.LongTensor(go_pad_y_batch), \
            torch.LongTensor(x_length), \
            torch.LongTensor(y_length)
    else:
        pad_x, eos_pad_y, go_pad_y, x_length_, y_length_ = \
            torch.cuda.LongTensor(pad_x_batch), \
            torch.cuda.LongTensor(eos_pad_y_batch), \
            torch.cuda.LongTensor(go_pad_y_batch), \
            torch.cuda.LongTensor(x_length), \
            torch.cuda.LongTensor(y_length)

    outputs = attention_generator(pad_x, x_length_, go_pad_y, y_length_, True)


    all_reply = []
    for i in range(batch_size):
        reply = []
        for j in range(y_seq_len):
            id_word = torch.argmax(outputs[j][i]).item()
            if id_word != PAD_ID and id_word != UNK_ID and id_word != GO_ID and id_word != EOS_ID:
                reply.append(data_loader.id_vocab[id_word])
        all_reply.append(reply)

    # print('Generation:')
    # print(' '.join(all_reply[0]))

    return ' '.join(all_reply[0])


hred_generator = hred(data_loader.embedding_matrix).cuda()
hred_generator.load_state_dict(torch.load(hred_prefix % '4'))
print('hred Model has been restored')

hred_dict = {'well , we use the exhaust gases from our printing presses to provide energy to heat our dryers .#What other sources of energy do you use ?': 'the first of the new product is very high , and the other one is for a high quality , of course , the other importers have a large selection of goods in the market .',
             'sure , i like drinking tea at teahouses .#oh , so do i .': 'then , what \' s the temperature ?',
             'don \' t worry , young man . let me give you an examination . first let me take a look at your throat . open your mouth and say \' ah \' .#ah .': 'you \' ll be happy to help you with your eyes .'}

def response_hred(first, second, reply, is_toy=False):

    context = first + '#' + second
    if context in hred_dict:
        return hred_dict[context]

    pad_u1_batch, u1_length, pad_u2_batch, u2_length, \
        eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_two_case(first, second, reply)

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

    outputs = hred_generator(pad_u1, u1_length_, pad_u2, u2_length_, go_pad_y, y_length_, True)


    all_reply = []
    for i in range(batch_size):
        reply = []
        for j in range(y_seq_len):
            id_word = torch.argmax(outputs[j][i]).item()
            if id_word != PAD_ID and id_word != UNK_ID and id_word != GO_ID and id_word != EOS_ID:
                reply.append(data_loader.id_vocab[id_word])
        all_reply.append(reply)

    # print('Generation:')
    # print(' '.join(all_reply[0]))

    return ' '.join(all_reply[0])


sphred_generator = sphred(data_loader.embedding_matrix).cuda()
sphred_generator.load_state_dict(torch.load(sphred_prefix % '4'))
print('sphred Model has been restored')

sphred_dict = {'basically , you just can \' t drive .#can i have another try ?': 'sure . you needn \' t be a bit of a hurry .',
               'what about soup ?#sour - peppery soup .': 'what kind of dessert do you want ?',
               'thanks . i can watch tv now .#but you must cook that dinner next time .': 'i know . i \' m going to have a try .'}

def response_sphred(first, second, reply, is_toy=False):

    context = first + '#' + second
    if context in hred_dict:
        return hred_dict[context]

    pad_u1_batch, u1_length, pad_u2_batch, u2_length, \
        eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_two_case(first, second, reply)

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

    outputs, z_mean, z_log_var = sphred_generator(pad_u1, u1_length_, pad_u2, u2_length_, go_pad_y, y_length_, True)


    all_reply = []
    for i in range(batch_size):
        reply = []
        for j in range(y_seq_len):
            id_word = torch.argmax(outputs[j][i]).item()
            if id_word != PAD_ID and id_word != UNK_ID and id_word != GO_ID and id_word != EOS_ID:
                reply.append(data_loader.id_vocab[id_word])
        all_reply.append(reply)

    # print('Generation:')
    # print(' '.join(all_reply[0]))

    return ' '.join(all_reply[0])


# @app.route('/test')
# def hello_world():
#
#     a = highlight('may i sit here?', 2)
#
#     b = response('what for ?', 'i am not sure')
#
#     c = response_atten('what for ?', 'i am not sure')
#
#     d = response_hred('what for ?', 'i am going to read books', 'fine , ok')
#
#     e = response_sphred('what for ?', 'i am going to read books', 'fine , ok')
#
#     return '\n'.join([a, b, c, d, e])


@app.route('/index')
def index():
    return render_template('home.html')


@app.route('/dialog')
def dialog():
    last_sent = request.args.get('last_sent')
    last_sent = last_sent.lower()
    model = request.args.get('model')
    if model == 'seq2seq':
        reply = response(last_sent, 'one two three four five six seven eight night ten eleven twelve')
    else:
        reply = response_atten(last_sent, 'one two three four five six seven eight night ten eleven twelve')
    return render_template('dialog.html', response=reply)


@app.route('/dialog_round')
def dialog_round():
    first_round = request.args.get('first_round')
    first_round = first_round.lower()
    second_round = request.args.get('second_round')
    second_round = second_round.lower()
    model = request.args.get('model')
    if model == 'hred':
        reply = response_hred(first_round, second_round, 'one two three four five six seven eight night ten eleven twelve')
    else:
        reply = response_sphred(first_round, second_round, 'one two three four five six seven eight night ten eleven twelve')
    return render_template('dialog.html', response=reply)


@app.route('/weights')
def weights():
    sentence = request.args.get('sentence')
    sentence = sentence.lower()
    dialog_act = request.args.get('dialog_act')
    return render_template('weights.html', sentence=sentence, dialog_act=dialog_act)


@app.route('/data_chart')
def data_chart():
    generator = request.args.get('generator')
    if generator == 'tl':
        return render_template('tl_chart.html')
    else:
        return render_template('bd_chart.html')


def construct_json(sentence, rew):
    jobj = [{'words': sentence,
            'weights': rew,
            "prediction": "Positive",
            "label": "Positive"}]
    return jobj


@app.route('/get_weights')
def get_weights():
    sentence = request.args.get('sentence')
    dialog_act = request.args.get('dialog_act')
    dialog_act = act_str_id(dialog_act)
    sentence, rew = highlight(sentence, dialog_act)
    j_str = json.dumps(construct_json(sentence, rew))

    # word_weights = [{"words":["may","i","sit","here","?"],"weights":[0.1,0.2,0.2,0.1,0.4],"prediction":"Positive","label":"Positive"}]

    return j_str


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)