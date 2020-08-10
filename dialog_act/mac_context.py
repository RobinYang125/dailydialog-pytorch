__author__ = 'yhd'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        else:
            self.source_train = 'data_root/dialogues_train.txt'
            self.source_test = 'data_root/dialogues_test.txt'
            self.batch_size = BATCH_SIZE
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = 'data_root/dialogues_validation.txt'
            self.test_batch_size = BATCH_SIZE
            self.val_batch_size = BATCH_SIZE

        # self.train_reader = textreader(self.source_train)
        # self.train_iterator = textiterator(self.train_reader, [self.batch_size, 2 * self.batch_size])
        #
        # self.test_reader = textreader(self.source_test)
        # self.test_iterator = textiterator(self.test_reader, [self.test_batch_size, 2 * self.test_batch_size])
        #
        # self.val_reader = textreader(self.source_validation)
        # self.val_iterator = textiterator(self.val_reader, [self.val_batch_size, 2 * self.val_batch_size])

        if platform.system() == 'Windows':
            with open(self.source_train, 'r', encoding='utf-8') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.source_validation, 'r', encoding='utf-8') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.source_test, 'r', encoding='utf-8') as stef:
                self.test_raw_text = stef.readlines()

        else:
            with open(self.source_train, 'r') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.source_validation, 'r') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.source_test, 'r') as stef:
                self.test_raw_text = stef.readlines()


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

    def dialogues_into_contexts(self, dialogues):
        qa_pairs = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')[:-1]
            if len(sentences) >= 3:
                if len(sentences) % 2 == 0:
                    sentences = sentences[:-1]
                qa_pairs.append([self.sentence_to_token_ids(sen) for sen in sentences])
        return qa_pairs

    def dialogues_into_qas_without_id(self, dialogues):
        qa_pairs = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')[:-1]
            if len(sentences) >= 3:
                for i in range(len(sentences) - 2):
                    qa = [sentences[i], sentences[i + 1], sentences[i + 2]]
                    qa_pairs.append(qa)
        return qa_pairs

    def dialogues_into_contexts_without_id(self, dialogues):
        qa_pairs = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')[:-1]
            if len(sentences) >= 3:
                if len(sentences) % 2 == 0:
                    sentences = sentences[:-1]
                qa_pairs.append(sentences)
        return qa_pairs

    def get_batch_test(self):
        if self.test_pointer < self.test_batch_num:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
        else:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size: ]

        self.test_pointer += 1

        qa_pairs = self.dialogues_into_contexts(raw_data)
        if len(qa_pairs) == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None])
        con_batch, y_batch = self.divide_sens(qa_pairs)
        con_batch = np.asarray(con_batch)

        self.test_raw_data = np.asarray(self.dialogues_into_contexts_without_id(raw_data))
        self.test_con_batch = self.test_raw_data[:, :-1]
        self.test_y_batch = self.test_raw_data[:, -1]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        return con_batch, self.add_eos(y_batch), self.add_go(y_batch), y_length

    def divide_sens(self, qa_pairs):
        con_batch = []
        y_batch = []
        for qa_batch in qa_pairs:
            con_batch.append(qa_batch[:-1])
            y_batch.append(qa_batch[-1])
        return con_batch, y_batch

    def get_batch_data(self):
        if self.train_pointer < self.batch_num:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
        else:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size: ]

        self.train_pointer += 1

        qa_pairs = self.dialogues_into_contexts(raw_data)
        if len(qa_pairs) == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None])
        con_batch, y_batch = self.divide_sens(qa_pairs)
        con_batch = np.asarray(con_batch)

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        return con_batch, self.add_eos(y_batch), self.add_go(y_batch), y_length

    def get_validation(self):
        if self.val_pointer < self.val_batch_num:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
        else:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size: ]

        self.val_pointer += 1

        qa_pairs = self.dialogues_into_contexts(raw_data)
        if len(qa_pairs) == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None])
        con_batch, y_batch = self.divide_sens(qa_pairs)
        con_batch = np.asarray(con_batch)

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        return con_batch, self.add_eos(y_batch), self.add_go(y_batch), y_length

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

# device = torch.device('cuda', 1)

import copy

# the first version
class Generator(nn.Module):
    def __init__(self, embeddings):
        super(Generator, self).__init__()
        self.hidden_size = EMBEDDING_SIZE
        self.output_size = VOCAB_SIZE
        self.q_dimension = 2 * self.hidden_size
        self.mid_rep = 4 * self.q_dimension
        self.mac_cell_number = 7
        self.drop_out = 0.2

        self.c_W_2dd = torch.randn((2 * self.q_dimension, self.q_dimension),
                                   requires_grad=True, dtype=torch.float32).cuda()

        self.c_b_d = torch.zeros((self.q_dimension, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.c_b_d = self.c_b_d.cuda()

        self.c_W_d1 = torch.randn((self.q_dimension, 1),
                                  requires_grad=True, dtype=torch.float32).cuda()

        self.c_b_1 = torch.zeros((1, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.c_b_1 = self.c_b_1.cuda()

        self.r_W_d_d_1 = torch.randn((self.q_dimension, self.q_dimension),
                                     requires_grad=True, dtype=torch.float32).cuda()

        self.r_b_d_1 = torch.zeros((self.q_dimension, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.r_b_d_1 = self.r_b_d_1.cuda()

        self.r_W_d_d_2 = torch.randn((self.q_dimension, self.q_dimension),
                                     requires_grad=True, dtype=torch.float32).cuda()

        self.r_b_d_2 = torch.zeros((self.q_dimension, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.r_b_d_2 = self.r_b_d_2.cuda()

        self.r_W_2d_d_3 = torch.randn((2 * self.q_dimension, self.q_dimension),
                                      requires_grad=True, dtype=torch.float32).cuda()

        self.r_b_d_3 = torch.zeros((self.q_dimension, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.r_b_d_3 = self.r_b_d_3.cuda()

        self.r_W_d_1_4 = torch.randn((self.q_dimension, 1),
                                     requires_grad=True, dtype=torch.float32).cuda()

        self.r_b_1_4 = torch.zeros((1, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.r_b_1_4 = self.r_b_1_4.cuda()

        self.w_W_2d_d_1 = torch.randn((2 * self.q_dimension, self.q_dimension),
                                      requires_grad=True, dtype=torch.float32).cuda()

        self.w_b_d_1 = torch.zeros((self.q_dimension, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.w_b_d_1 = self.w_b_d_1.cuda()

        self.w_W_d_1_2 = torch.randn((self.q_dimension, 1),
                                     requires_grad=True, dtype=torch.float32).cuda()

        self.w_b_1_2 = torch.zeros((1, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.w_b_1_2 = self.w_b_1_2.cuda()

        self.w_W_2d_d_3 = torch.randn((2 * self.q_dimension, self.q_dimension),
                                      requires_grad=True, dtype=torch.float32).cuda()

        self.w_b_d_3 = torch.zeros((self.q_dimension, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.w_b_d_3 = self.w_b_d_3.cuda()

        self.w_W_d_d_4 = torch.randn((self.q_dimension, self.q_dimension),
                                     requires_grad=True, dtype=torch.float32).cuda()

        self.w_b_d_4 = torch.zeros((self.q_dimension, ), requires_grad=True, dtype=torch.float32) + 0.1
        self.w_b_d_4 = self.w_b_d_4.cuda()

        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.kb_encoder = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True)
        self.q_encoder = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True)

        self.state_layer = nn.Sequential(nn.Linear(self.mid_rep, self.hidden_size),
                                         nn.Dropout(self.drop_out),)

        self.q_layer = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                         nn.Dropout(self.drop_out),)

        self.decoder = nn.GRU(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, con_input, target_input):
        initial_m = torch.randn((1, self.q_dimension), requires_grad=True, dtype=torch.float32).cuda()
        all_m_list = [initial_m]
        initial_c = torch.randn((1, self.q_dimension), requires_grad=True, dtype=torch.float32).cuda()
        all_c_list = [initial_c]

        for i in range(len(con_input) / 2):
            kb_embed = self.embedding_layer(con_input[2 * i])
            q_embed = self.embedding_layer(con_input[2 * i + 1])

            # print('kb_embed', kb_embed)
            # print('q_embed', q_embed)

            kb_hidden = None
            kb_outputs = []
            kb_states = []
            for idx, word in enumerate(kb_embed):
                kb_output, kb_hidden = self.kb_encoder(word.view(1, 1, -1), kb_hidden)
                kb_outputs.append(kb_output)
                kb_states.append(kb_hidden)

            # [1, 29, 600]
            kb = torch.stack(kb_outputs, dim=1).view(1, -1, self.hidden_size * 2)

            q_hidden = None
            q_outputs = []
            q_states = []
            for idx, word in enumerate(q_embed):
                q_output, q_hidden = self.kb_encoder(word.view(1, 1, -1), q_hidden)
                q_outputs.append(q_output)
                q_states.append(q_hidden)

            # [1, 29, 600]
            cw = torch.stack(q_outputs, dim=1).view(1, -1, self.hidden_size * 2)
            # [1, 1200]
            q = torch.cat((self.q_layer(q_outputs[0]), self.q_layer(q_outputs[-1])), dim=-1).view(1, -1)

            for _ in range(self.mac_cell_number):
                c_i, m_i = self.mac_cell(all_c_list[-1], all_m_list[-1], q, cw, kb, all_m_list, all_c_list)
                all_m_list.append(m_i)
                all_c_list.append(c_i)

        # print('q', q)
        # print('all_m_list[-1]', all_m_list[-1])
        reasoning_output = F.tanh(torch.cat((q, all_m_list[-1]), dim=-1))
        # print('reasoning_output', reasoning_output)
        # print('kb_states[0].view(1, -1)', kb_states[0].view(1, -1))
        # print('q_states[0].view(1, -1)', q_states[0].view(1, -1))
        middle_representation = torch.cat((kb_states[0].view(1, -1), q_states[0].view(1, -1), reasoning_output), dim=-1)
        # print('middle_representation', middle_representation)
        decoder_hidden = self.state_layer(middle_representation).view(1, 1, -1)
        # print('decoder_hidden', decoder_hidden)
        target_embed = self.embedding_layer(target_input)

        # emb = self.embedding_layer(torch.cuda.LongTensor([GO_ID]))
        outputs = []
        for idx, word in enumerate(target_embed):
            output, decoder_hidden = self.decoder(word.view(1, 1, -1), decoder_hidden)
            outputs.append(self.output_layer(output[0]))

            # output, hidden = self.docoder(emb.view(1, 1, -1), hidden)
            # output_vocab = self.output_layer(output[0])
            # output_id = torch.argmax(output_vocab)
            # emb = self.embedding_layer(output_id)
            # if output_id.item() == EOS_ID:
            #     break
            # outputs.append(output_vocab)
        return outputs


    def mac_cell(self, last_c, last_m, q, cw, kb, all_last_m, all_last_c):
        c_i = self.control_unit(last_c, q, cw)
        m_new = self.read_unit(last_m, kb, c_i)
        m_i = self.write_unit(m_new, last_m, all_last_m, c_i, all_last_c)
        return c_i, m_i

    #                        [1, 1200]     [1, 1200] [1, 29, 1200]
    def control_unit(self, last_cell_state, query, c_w):
        c_W_dd_i = torch.randn((self.q_dimension, self.q_dimension), requires_grad=True, dtype=torch.float32).cuda()
        c_b_d_i = torch.zeros((self.q_dimension, ), requires_grad=True, dtype=torch.float32) + 0.1
        c_b_d_i = c_b_d_i.cuda()

        # [1, 1200]
        q_i = torch.matmul(query, c_W_dd_i) + c_b_d_i
        q_i = F.tanh(q_i)

        # [1, 1200]
        cq_i = torch.matmul(
                    torch.cat((q_i, last_cell_state), dim=-1),
                    self.c_W_2dd
                ) + self.c_b_d
        cq_i = F.tanh(cq_i)

        # [1, 1200]
        c_i = torch.mean(
                F.softmax(
                    #           [1, 1200]   [1, 29, 600]
                    torch.matmul(cq_i * c_w, self.c_W_d1) + self.c_b_1, dim=1
                ).view(1, -1, 1) * c_w, dim=1
            )
        c_i = F.tanh(c_i)

        return c_i

    #                [1, 1200]  [1, 29, 1200] [1, 1200]
    def read_unit(self, last_m, kb, c_i):

        # [1, 1200]
        m_ = torch.matmul(last_m, self.r_W_d_d_1) + self.r_b_d_1
        m_ = F.tanh(m_)

        # [1, 29, 1200]
        kb_ = torch.matmul(kb, self.r_W_d_d_2) + self.r_b_d_2
        kb_ = F.tanh(kb_)

        # [1, 29, 1200]
        i_m_kb = kb_ * m_

        # [1, 29, 1200]
        i_m_kb_ = torch.matmul(torch.cat((i_m_kb, kb), dim=-1), self.r_W_2d_d_3) + self.r_b_d_3
        i_m_kb_ = F.tanh(i_m_kb_)

        # [1, 29, 1200]
        c_i_m_kb = i_m_kb_ * c_i

        # [1, 1200]
        m_new = torch.mean(
                    F.softmax(
                        torch.matmul(c_i_m_kb, self.r_W_d_1_4) + self.r_b_1_4, dim=1
                    ).view(1, -1, 1) * kb, dim=1
                )
        m_new = F.tanh(m_new)

        return m_new

    #                 [1, 1200] [1, 1200]  list             list
    def write_unit(self, m_new, last_m, all_last_m, c_i, all_last_c):
        all_last_m_ = torch.stack(all_last_m).view(1, -1, self.q_dimension)
        # print('all_last_m_', all_last_m_)

        all_last_c_ = torch.stack(all_last_c).view(1, -1, self.q_dimension)
        # print('all_last_c_', all_last_c_)

        # [1, 1200]
        m_ = torch.matmul(
                        torch.cat((m_new, last_m), dim=-1), self.w_W_2d_d_1
             ) + self.w_b_d_1
        m_ = F.tanh(m_)
        # print('m_', m_)

        # [1, 1200]
        m_sa = torch.mean(
                    F.softmax(
                        torch.matmul(
                            # [1, j-1, d] [1, 1200]
                            all_last_c_ * c_i,
                            self.w_W_d_1_2
                        ) + self.w_b_1_2, dim=1
                    ).view(1, -1, 1) * all_last_m_, dim=1
                )
        m_sa = F.tanh(m_sa)
        # print('m_sa', m_sa)

        # [1, 1200]
        m__ = torch.matmul(torch.cat((m_sa, m_), dim=-1), self.w_W_2d_d_3) + self.w_b_d_3
        m__ = F.tanh(m__)
        # print('m__', m__)

        # [1, 1200]
        c_ = torch.matmul(c_i, self.w_W_d_d_4) + self.w_b_d_4
        c_ = F.tanh(c_)
        # print('c_', c_)

        # [1, 1200]
        m_i = F.sigmoid(c_) * last_m + (1 - F.sigmoid(c_)) * m__
        # print('m_i', m_i)

        return m_i


EPOCH_SIZE = 50

prefix = 'mac_context'

act_prefix = 'models_adan_act/adan_para_%s.pkl'
emotion_prefix = 'models_adan_emotion/adan_para_%s.pkl'

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
    #     act.load_state_dict(torch.load(act_prefix%'10'))
    # else:
    #     act = DAN_ACT(data_loader.embedding_matrix).cuda()
    #     act.load_state_dict(torch.load(act_prefix%'50'))
    #
    # if is_toy:
    #     emotion = DAN_EMOTION(data_loader.embedding_matrix)
    #     emotion.load_state_dict(torch.load(emotion_prefix%'10'))
    # else:
    #     emotion = DAN_EMOTION(data_loader.embedding_matrix).cuda()
    #     emotion.load_state_dict(torch.load(emotion_prefix%'50'))

    if is_toy:
        generator = Generator(data_loader.embedding_matrix)
        # generator.load_state_dict(torch.load(model_prefix%'2'))
    else:
        generator = Generator(data_loader.embedding_matrix).cuda()
        # generator.load_state_dict(torch.load(model_prefix%'7'))
        # torch.backends.cudnn.enabled = False
        # print('Model has been loaded')
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

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
            con_batch, eos_y_batch, go_y_batch, y_length = data_loader.get_batch_data()

            if con_batch.all() == None:
                continue

            for idx, y_len_ele in enumerate(y_length):

                # if not is_toy:
                #     print('train: %d batch and %d sentence' % (bn, idx))

                loss_mean = 0

                if is_toy:
                    con_, eos_pad_y, go_pad_y = \
                        torch.LongTensor(con_batch[idx]), \
                        torch.LongTensor(eos_y_batch[idx]), \
                        torch.LongTensor(go_y_batch[idx])
                else:
                    # con_, eos_pad_y, go_pad_y = \
                    #     torch.cuda.LongTensor(con_batch[idx]), \
                    #     torch.cuda.LongTensor(eos_y_batch[idx]), \
                    #     torch.cuda.LongTensor(go_y_batch[idx])

                    eos_pad_y, go_pad_y = \
                        torch.cuda.LongTensor(eos_y_batch[idx]), \
                        torch.cuda.LongTensor(go_y_batch[idx])
                    con_ = [torch.cuda.LongTensor(item) for item in con_batch[idx]]

                step += 1

                outputs = generator(con_, go_pad_y)

                # pred_ids = []
                # for jdx, ele in enumerate(outputs):
                #     pred_ids.append(torch.argmax(ele))
                # pred_id_tensor = torch.stack(pred_ids).view(1, -1)
                #
                # # act
                # act_outputs, act_rewards = act(pred_id_tensor)
                # cal_label = torch.argmax(act_outputs, dim=-1).item()
                #
                # if cal_label == y_act_batch[idx]:
                #     final_act_rewards = act_rewards
                # else:
                #     final_act_rewards = 1 - act_rewards
                #
                #
                # # emotion
                # emotion_outputs, emotion_rewards = emotion(pred_id_tensor)
                # cal_label = torch.argmax(emotion_outputs, dim=-1).item()
                #
                # if cal_label == y_emotion_batch[idx]:
                #     final_emotion_rewards = emotion_rewards
                # else:
                #     final_emotion_rewards = 1 - emotion_rewards
                #
                # final_act_rewards, final_emotion_rewards = final_act_rewards[0], final_emotion_rewards[0]

                for jdx, words in enumerate(outputs):
                    # loss_mean += loss_func(outputs[jdx], eos_pad_y[jdx].view(1,)) \
                    #              * final_act_rewards[jdx] * final_emotion_rewards[jdx]
                    loss_mean += loss_func(outputs[jdx], eos_pad_y[jdx].view(1,))
                    print('loss_mean', loss_mean)

                generator_optimizer.zero_grad()
                loss_mean.backward()
                generator_optimizer.step()

                losses += loss_mean.item() / y_length[idx]

        generator.eval()

        for bn in range(data_loader.val_batch_num + 1):
            con_batch, eos_y_batch, go_y_batch, y_length = data_loader.get_validation()

            if con_batch.all() == None:
                continue

            for idx, y_len_ele in enumerate(y_length):

                # if not is_toy:
                #     print('validation: %d batch and %d sentence' % (bn, idx))

                val_loss_mean = 0

                if is_toy:
                    con_, eos_pad_y, go_pad_y = \
                        torch.LongTensor(con_batch[idx]), \
                        torch.LongTensor(eos_y_batch[idx]), \
                        torch.LongTensor(go_y_batch[idx])
                else:
                    eos_pad_y, go_pad_y = \
                        torch.cuda.LongTensor(eos_y_batch[idx]), \
                        torch.cuda.LongTensor(go_y_batch[idx])
                    con_ = [torch.cuda.LongTensor(item) for item in con_batch[idx]]

                outputs = generator(con_, go_pad_y)


                # pred_ids = []
                # for jdx, ele in enumerate(outputs):
                #     pred_ids.append(torch.argmax(ele))
                # pred_id_tensor = torch.stack(pred_ids).view(1, -1)
                #
                # # act
                # act_outputs, act_rewards = act(pred_id_tensor)
                # cal_label = torch.argmax(act_outputs, dim=-1).item()
                #
                # if cal_label == y_act_batch[idx]:
                #     final_act_rewards = act_rewards
                # else:
                #     final_act_rewards = 1 - act_rewards
                #
                # # emotion
                # emotion_outputs, emotion_rewards = emotion(pred_id_tensor)
                # cal_label = torch.argmax(emotion_outputs, dim=-1).item()
                #
                # if cal_label == y_emotion_batch[idx]:
                #     final_emotion_rewards = emotion_rewards
                # else:
                #     final_emotion_rewards = 1 - emotion_rewards
                #
                # final_act_rewards, final_emotion_rewards = final_act_rewards[0], final_emotion_rewards[0]


                for jdx, words in enumerate(outputs):
                    # val_loss_mean += loss_func(outputs[jdx], eos_pad_y[jdx].view(1,)) \
                    #              * final_act_rewards[jdx] * final_emotion_rewards[jdx]
                    val_loss_mean += loss_func(outputs[jdx], eos_pad_y[jdx].view(1,))

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

if __name__ == '__main__':
    main_train(True)

    # main_test(True)

    # main_train()

    # main_test()

    # d = DataLoader()
    #
    # for bn in range(d.batch_num + 1):
    #     res = d.get_batch_data()
    #     print(res[0])
    #     print(res[1])
    #     print(res[2])