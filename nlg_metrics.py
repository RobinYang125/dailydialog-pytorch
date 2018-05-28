__author__ = 'yhd'

# from nlgeval import compute_individual_metrics
# metrics_dict = compute_individual_metrics(manual_summmary, system_generated_summary)
# print(metrics_dict)

bf_reply = 'metrics/bf_reply.txt'
bf_answer = 'metrics/bf_answer.txt'
bf_scores = 'metrics/bf_scores.txt'

seq2seq_reply = 'metrics/seq2seq_reply.txt'
seq2seq_answer = 'metrics/seq2seq_answer.txt'
seq2seq_scores = 'metrics/seq2seq_scores.txt'

mac_reply = 'metrics/mac_reply.txt'
mac_answer = 'metrics/mac_answer.txt'
mac_scores = 'metrics/mac_scores.txt'

d_dan_reply = 'metrics/d_dan_reply.txt'
d_dan_answer = 'metrics/d_dan_answer.txt'
d_dan_scores = 'metrics/d_dan_scores.txt'

d_dan_p_reply = 'metrics/d_dan_p_reply.txt'
d_dan_p_answer = 'metrics/d_dan_p_answer.txt'
d_dan_p_scores = 'metrics/d_dan_p_scores.txt'

vpn_end_to_end_reply = 'metrics/vpn_end_to_end_reply.txt'
vpn_end_to_end_answer = 'metrics/vpn_end_to_end_answer.txt'
vpn_end_to_end_scores = 'metrics/vpn_end_to_end_scores.txt'

vpn_end_to_end_p_reply = 'metrics/vpn_end_to_end_p_reply.txt'
vpn_end_to_end_p_answer = 'metrics/vpn_end_to_end_p_answer.txt'
vpn_end_to_end_p_scores = 'metrics/vpn_end_to_end_p_scores.txt'

vpn_rl_reply = 'metrics/vpn_rl_reply.txt'
vpn_rl_answer = 'metrics/vpn_rl_answer.txt'
vpn_rl_scores = 'metrics/vpn_rl_scores.txt'

seq_torch_seq_reply = 'metrics/seq_torch_seq_reply.txt'
seq_torch_seq_answer = 'metrics/seq_torch_seq_answer.txt'
seq_torch_seq_scores = 'metrics/seq_torch_seq_scores.txt'

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def get_all_scores(reply, answer, scores):
    from nlgeval import compute_metrics
    metrics_dict = compute_metrics(hypothesis=reply,
                                   references=[answer])

    print(metrics_dict)

    import codecs
    with codecs.open(scores, 'w', encoding='utf-8') as fo:
        fo.write(str(metrics_dict))

if __name__ == '__main__':
    get_all_scores(seq_torch_seq_reply, seq_torch_seq_answer, seq_torch_seq_scores)

