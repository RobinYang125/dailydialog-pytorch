__author__ = 'yhd'

import os
import re

from tensorflow.python.platform import gfile
from collections import defaultdict
import jieba.posseg as pseg
import nltk

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")

def get_dialog_train_set_path(path):
  return os.path.join(path, 'train')


def get_dialog_dev_set_path(path):
  return os.path.join(path, 'test')

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w.lower() for w in words if w]

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    #with gfile.GFile(data_path, mode="r") as f:
    with open(data_path,'rb') as f:
      counter = 0
      for line in f.readlines():
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line.decode('utf-8')) if tokenizer else basic_tokenizer(line.decode('utf-8'))
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")

def initialize_vocabulary(vocabulary_path):
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
    return vocab, rev_vocab

'''
Chinese

def build_ntable():
    d = defaultdict(int)
    f = open('data_root/train.txt', 'rb')
    f1 = open('data_root/nouns2500.in', 'w')
    s = f.readline()
    while s:
        words = pseg.cut(s.strip())
        for ele in words:
            print(ele)
            if ele.flag[0]=='n':
                d[ele.word] += 1
        s = f.readline()
    nl = d.items()
    nl = sorted(nl,key=lambda x:x[1],reverse=True)
    for ele in nl[:2500]:
        f1.write(ele[0]+'\n')
    f.close()
    f1.close()
'''

def build_ntable():
    d = defaultdict(int)
    f = open('data_root/dialogues_text.txt', 'r', encoding='utf-8')
    f1 = open('data_root/nouns2500.in', 'w', encoding='utf-8')
    s = f.readline()
    while s:
        tokens = nltk.word_tokenize(s.strip())
        words = nltk.pos_tag(tokens)
        for word,pos in words:
            if pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS':
                d[word] += 1
        s = f.readline()
    nl = d.items()
    nl = sorted(nl,key=lambda x:x[1],reverse=True)
    for ele in nl[:2500]:
        f1.write(ele[0]+'\n')
    f.close()
    f1.close()

def build_noun():
    d = defaultdict(int)
    f = open('data_root/dialogues_text_without_eou.txt', 'r', encoding='utf-8')
    f1 = open('data_root/nouns20000.in', 'w', encoding='utf-8')
    s = f.readline()
    while s:
        tokens = nltk.word_tokenize(s.strip())
        words = nltk.pos_tag(tokens)
        for word,pos in words:
            if pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS':
                d[word] += 1
        s = f.readline()
    nl = d.items()
    nl = sorted(nl,key=lambda x:x[1],reverse=True)
    for ele in nl[:20000]:
        f1.write(ele[0]+'\n')
    f.close()
    f1.close()

def build_verb():
    d = defaultdict(int)
    f = open('data_root/dialogues_text_without_eou.txt', 'r', encoding='utf-8')
    f1 = open('data_root/verbs10000.in', 'w', encoding='utf-8')
    s = f.readline()
    while s:
        tokens = nltk.word_tokenize(s.strip())
        words = nltk.pos_tag(tokens)
        for word,pos in words:
            if pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' \
                    or pos == 'VBP' or pos == 'VBZ':
                d[word] += 1
        s = f.readline()
    nl = d.items()
    nl = sorted(nl,key=lambda x:x[1],reverse=True)
    for ele in nl[:10000]:
        f1.write(ele[0]+'\n')
    f.close()
    f1.close()

def build_pron():
    d = defaultdict(int)
    f = open('data_root/dialogues_text_without_eou.txt', 'r', encoding='utf-8')
    f1 = open('data_root/prons10000.in', 'w', encoding='utf-8')
    s = f.readline()
    while s:
        tokens = nltk.word_tokenize(s.strip())
        words = nltk.pos_tag(tokens)
        for word,pos in words:
            if pos == 'PRP' or pos == 'PRP$':
                d[word] += 1
        s = f.readline()
    nl = d.items()
    nl = sorted(nl,key=lambda x:x[1],reverse=True)
    for ele in nl[:10000]:
        f1.write(ele[0]+'\n')
    f.close()
    f1.close()

def squeeze_sentence(raw_text, vocab_file, output_path):
    vocab, rev_vocab = initialize_vocabulary(vocab_file)
    with open(raw_text, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    with open(output_path, 'w', encoding='utf-8') as o:
        for idx, line in enumerate(raw_data):
            dialogs = line.split('__eou__')[:-1]
            for sentence in dialogs:
                words = sentence.split()
                for word in words:
                    if word in rev_vocab:
                        o.write(word + ' ')
                o.write('__eou__')
            o.write('\n')


def sentence_to_token_ids(sentence):
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
  words = basic_tokenizer(sentence)
  # Normalize digits by 0 before looking words up in the vocabulary.
  words_new = [re.sub(_DIGIT_RE, "0", w) for w in words]
  return ' '.join(words_new)

def tag_sentence(raw_text, output_path):
    nouns_vocab, nouns_rev_vocab = initialize_vocabulary('data_root/nouns10000.in')
    verbs_vocab, verbs_rev_vocab = initialize_vocabulary('data_root/verbs10000.in')
    prons_vocab, prons_rev_vocab = initialize_vocabulary('data_root/prons10000.in')
    with open(raw_text, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    # N V P S
    # 0 1 2 3
    with open(output_path, 'w', encoding='utf-8') as o:
        for idx, line in enumerate(raw_data):
            dialogs = line.split('__eou__')[:-1]
            for sentence in dialogs:
                words = sentence_to_token_ids(sentence)
                for word in words:
                    if word in nouns_rev_vocab:
                        # o.write('N' + ' ')
                        o.write('0' + ' ')
                    elif word in verbs_rev_vocab:
                        # o.write('V' + ' ')
                        o.write('1' + ' ')
                    elif word in prons_rev_vocab:
                        # o.write('P' + ' ')
                        o.write('2' + ' ')
                    else:
                        # o.write('S' + ' ')
                        o.write('3' + ' ')
                o.write('__eou__')
            o.write('\n')

def test_pos_tag():
    words = nltk.word_tokenize('I need this tool to meet my needs')
    print(words)
    word_tag = nltk.pos_tag(words)
    print(word_tag)


def tag_true_sentence(raw_text, output_path):
    with open(raw_text, 'r') as f:
        raw_data = f.readlines()

    # N V P S
    # 0 1 2 3
    with open(output_path, 'w') as o:
        for line in raw_data:
            sentences = line.split('__eou__')[:-1]
            for sentence in sentences:
                new_sentence = sentence_to_token_ids(sentence)
                words = nltk.word_tokenize(new_sentence)
                word_tag = nltk.pos_tag(words)
                for word,pos in word_tag:
                    if pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS':
                        o.write('0 ')
                    if pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' \
                            or pos == 'VBP' or pos == 'VBZ':
                        o.write('1 ')
                    elif pos == 'PRP' or pos == 'PRP$':
                        o.write('2 ')
                    else:
                        o.write('3 ')
                o.write('__eou__ ')
            o.write('\n')

if __name__ == '__main__':

    # test_pos_tag()
    # res = sentence_to_token_ids('I need this tool to meet my needs')
    # print(res)
    # build_noun()

    # squeeze_sentence('data_root/dialogues_test.txt',
    #                  'data_root/nouns20000.in', 'data_root/words/dialogues_test_nouns.txt')
    # squeeze_sentence('data_root/dialogues_validation.txt',
    #                  'data_root/nouns20000.in', 'data_root/words/dialogues_validation_nouns.txt')
    # squeeze_sentence('data_root/dialogues_train.txt',
    #                  'data_root/nouns20000.in', 'data_root/words/dialogues_train_nouns.txt')

    # squeeze_sentence('data_root/dialogues_test.txt',
    #                  'data_root/verbs10000.in', 'data_root/words/dialogues_test_verbs.txt')
    # print('1')
    # squeeze_sentence('data_root/dialogues_validation.txt',
    #                  'data_root/verbs10000.in', 'data_root/words/dialogues_validation_verbs.txt')
    # print('2')
    # squeeze_sentence('data_root/dialogues_train.txt',
    #                  'data_root/verbs10000.in', 'data_root/words/dialogues_train_verbs.txt')
    # print('3')
    #
    # squeeze_sentence('data_root/dialogues_test.txt',
    #                  'data_root/prons10000.in', 'data_root/words/dialogues_test_prons.txt')
    # print('4')
    # squeeze_sentence('data_root/dialogues_validation.txt',
    #                  'data_root/prons10000.in', 'data_root/words/dialogues_validation_prons.txt')
    # print('5')
    # squeeze_sentence('data_root/dialogues_train.txt',
    #                  'data_root/prons10000.in', 'data_root/words/dialogues_train_prons.txt')
    # print('6')

    # tag_sentence('data_root/dialogues_train.txt',  'data_root/words/dialogues_train_tag.txt')
    # tag_sentence('data_root/dialogues_validation.txt',  'data_root/words/dialogues_validation_tag.txt')
    # tag_sentence('data_root/dialogues_test.txt',  'data_root/words/dialogues_test_tag.txt')

    # tag_true_sentence('data_root/dialogues_train.txt',  'data_root/words/dialogues_train_tag.txt')
    tag_true_sentence('data_root/dialogues_validation.txt',  'data_root/words/dialogues_validation_tag.txt')
    tag_true_sentence('data_root/dialogues_test.txt',  'data_root/words/dialogues_test_tag.txt')


# import pickle
#
# def build_wqtable():
#     d = defaultdict(int)
#     f = open('data_root/dialogues_text.txt', 'r', encoding='utf-8')
#     s = f.readline()
#     while s:
#         s = s.decode('utf-8').replace(' ','')
#         s = ' '.join(list(s)).encode('utf-8')
#         s = s.split()
#         for w in s:
#             d[w]+=1
#         f.readline()
#         s = f.readline()
#     f.close()
#     f = open('data_root/q_table.pkl','w')
#     total = sum(d.values())
#     d = dict([(k,1.*v/total) for k,v in d.items()])
#     pickle.dump(d,f)
#     f.close()

# def build_table(data,vocab,nouns):
#     tmp = []
#     for w in vocab:
#         for w_ in nouns:
#             tmp.append(w+'|'+w_)
#     cotable = dict([(ele,1) for ele in tmp])
#     #ntable = dict([(ele,1) for ele in nouns])
#     #qtable = dict([(ele,1) for ele in vocab])
#     for ele in data:
#         q,a = ele
#         aws = a.strip().split()
#         qws = list(q.decode('utf-8').replace(' ',''))
#         qws = ' '.join(qws)
#         qws = qws.encode('utf-8')
#         #print('%d:%s'%(i,qws))
#         qws = qws.split()
#         for ind,w in enumerate(aws):
#             for w_ in qws:
#                 try:
#                     cotable[w_+'|'+w] += 1
#                 except:
#                     pass
#             #try:
#                # ntable[w] += 1
#             #except:
#                # pass
#     #total = sum(cotable.values())
#     ntable = defaultdict(int)
#     qtable = defaultdict(int)
#     # k is q|r
#     for k,ele in cotable.items():
#         # r
#         ntable[k.split('|')[1]] += ele
#         # q
#         qtable[k.split('|')[0]] += ele
#
#     cotable = dict([(k,1.*v/ntable[k.split('|')[1]]) for k,v in cotable.items()])
#     total = sum(qtable.values())
#     qtable = dict([(k,1.*v/total) for k,v in qtable.items()])
#     f = open('co_table.pkl','w')
#     pickle.dump(cotable,f)
#     f.close()
#     f = open('n_table.pkl','w')
#     pickle.dump(ntable,f)
#     f.close()
#     f = open('q_table.pkl','w')
#     pickle.dump(qtable,f)
#     f.close()
#
# def process_for_table(data_path='data_root/dialogues_test.txt',
#                       vocab_path='data_root/vocab50000.in', nouns_path='data_root/vocab50000.in'):
#     with open(data_path, 'r') as df:
#         data = []
#         for line in df.readlines():
#             qas = line.split('__eou__')
#             for i in range(len(qas) - 2):
#                 data.append((qas[i], qas[i + 1]))
#     with open(vocab_path, 'r') as df:
#         vocab = df.readlines()
#         vocab = [ele.strip() for ele in vocab]
#     with open(nouns_path, 'r') as df:
#         nouns = df.readlines()
#         nouns = [ele.strip() for ele in nouns]
#
#     build_table(data, vocab, nouns)



# if __name__ == '__main__':
#
#     # data_dir = 'data_root'
#     # vocabulary_size = 200
#     #
#     # train_path = get_dialog_train_set_path(data_dir)
#     # vocab_path = os.path.join(data_dir, "vocab%d.in" % vocabulary_size)
#     #
#     # # 19495 lines
#     # create_vocabulary(vocab_path, train_path + ".txt", vocabulary_size)
#     #
#     # # print(initialize_vocabulary(vocab_path))
#     #
#     # # build_ntable()
#
#     # process_for_table()
#
#     # build_pron()

