__author__ = 'yhd'

import codecs

import sys
reload(sys)
sys.setdefaultencoding('utf8')

taja_path = 'results/taja_w_bs_results.txt'
taja_reply_path = 'metrics/taja_reply.txt'
taja_answer_path = 'metrics/taja_answer.txt'

mac_reasoning_path = 'results/mac_reason_results.txt'
mac_reply_path = 'metrics/mac_reply.txt'
mac_answer_path = 'metrics/mac_answer.txt'

seq_reasoning_path = 'models_seq2seq/results.txt'
seq_reply_path = 'metrics/seq_reply.txt'
seq_answer_path = 'metrics/seq_answer.txt'

def from_txt_get_reply_answer(path, reply_path, answer_path):
    with codecs.open(path, 'r', encoding='utf-8') as f_input:
        with codecs.open(reply_path, 'w', encoding='utf-8') as f_reply:
            with codecs.open(answer_path, 'w', encoding='utf-8') as f_answer:
                raw_data = f_input.readlines()
                for idx, line in enumerate(raw_data):
                    if 'Answer:' in line:
                        f_answer.write(raw_data[idx + 1])
                    elif 'Generation:' in line:
                        reply = raw_data[idx + 1].split()
                        true_reply = [item for item in reply if item != '_EOS']
                        f_reply.write(' '.join(true_reply) + '\n')


def taja_from_txt_get_reply_answer(path=taja_path, reply_path=taja_reply_path, answer_path=taja_answer_path):
    with codecs.open(path, 'r', encoding='utf-8') as f_input:
        with codecs.open(reply_path, 'w', encoding='utf-8') as f_reply:
            with codecs.open(answer_path, 'w', encoding='utf-8') as f_answer:
                raw_data = f_input.readlines()
                counter = 1
                for idx, line in enumerate(raw_data):
                    if 'Answer:' in line:
                        f_answer.write(raw_data[idx + 1])
                    elif 'Question:' in line and counter == 2:
                        reply = raw_data[idx + 1].split()
                        true_reply = [item for item in reply if item != '_EOS']
                        f_reply.write(' '.join(true_reply) + '\n')
                        counter = 1
                    elif 'Question:' in line:
                        counter += 1

if __name__ == '__main__':
    from_txt_get_reply_answer(seq_reasoning_path, seq_reply_path, seq_answer_path)
    # taja_from_txt_get_reply_answer()
