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

seq_torch_seq_sgd_reply = 'metrics/seq_torch_seq_sgd_reply.txt'
seq_torch_seq_sgd_answer = 'metrics/seq_torch_seq_sgd_answer.txt'
seq_torch_seq_sgd_scores = 'metrics/seq_torch_seq_sgd_scores.txt'

seq_dan_wsr_reply = 'metrics/seq_dan_wsr_reply.txt'
seq_dan_wsr_answer = 'metrics/seq_dan_wsr_answer.txt'
seq_dan_wsr_scores = 'metrics/seq_dan_wsr_scores.txt'

dan_wr_reply = 'metrics/dan_wr_reply.txt'
dan_wr_answer = 'metrics/dan_wr_answer.txt'
dan_wr_scores = 'metrics/dan_wr_scores.txt'

seq_torch_batch_reply = 'metrics/seq_torch_batch_reply.txt'
seq_torch_batch_answer = 'metrics/seq_torch_batch_answer.txt'
seq_torch_batch_scores = 'metrics/seq_torch_batch_scores.txt'

dan_wr_batch_reply = 'metrics/seq_dan_wsr_reply.txt'
dan_wr_batch_answer = 'metrics/seq_dan_wsr_answer.txt'
dan_wr_batch_scores = 'metrics/seq_dan_wsr_scores.txt'

seq_attention_torch_reply = 'metrics/seq_attention_torch_reply.txt'
seq_attention_torch_answer = 'metrics/seq_attention_torch_answer.txt'
seq_attention_torch_scores = 'metrics/seq_attention_torch_scores.txt'

dan_seq_atten_reply = 'metrics/dan_seq_atten_reply.txt'
dan_seq_atten_answer = 'metrics/dan_seq_atten_answer.txt'
dan_seq_atten_scores = 'metrics/dan_seq_atten_scores.txt'

dan_seq_atten_50_reply = 'metrics/dan_seq_atten_50_reply.txt'
dan_seq_atten_50_answer = 'metrics/dan_seq_atten_50_answer.txt'
dan_seq_atten_50_scores = 'metrics/dan_seq_atten_50_scores.txt'

dan_seq_atten_21_reply = 'metrics/dan_seq_atten_21_reply.txt'
dan_seq_atten_21_answer = 'metrics/dan_seq_atten_21_answer.txt'
dan_seq_atten_21_scores = 'metrics/dan_seq_atten_21_scores.txt'

hred_torch_reply = 'metrics/hred_torch_reply.txt'
hred_torch_answer = 'metrics/hred_torch_answer.txt'
hred_torch_scores = 'metrics/hred_torch_scores.txt'

hred_torch_3_reply = 'metrics/hred_torch_3_reply.txt'
hred_torch_3_answer = 'metrics/hred_torch_3_answer.txt'
hred_torch_3_scores = 'metrics/hred_torch_3_scores.txt'

hred_torch_4_reply = 'metrics/hred_torch_4_reply.txt'
hred_torch_4_answer = 'metrics/hred_torch_4_answer.txt'
hred_torch_4_scores = 'metrics/hred_torch_4_scores.txt'

hred_dan_reply = 'metrics/hred_dan_reply.txt'
hred_dan_answer = 'metrics/hred_dan_answer.txt'
hred_dan_scores = 'metrics/hred_dan_scores.txt'

hred_dan_batch_reply = 'metrics/hred_dan_batch_12_reply.txt'
hred_dan_batch_answer = 'metrics/hred_dan_batch_12_answer.txt'
hred_dan_batch_scores = 'metrics/hred_dan_batch_12_scores.txt'

hred_dan_batch_12_dan_50_reply = 'metrics/hred_dan_batch_12_dan_50_reply.txt'
hred_dan_batch_12_dan_50_answer = 'metrics/hred_dan_batch_12_dan_50_answer.txt'
hred_dan_batch_12_dan_50_scores = 'metrics/hred_dan_batch_12_dan_50_scores.txt'


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
    get_all_scores(dan_seq_atten_21_reply, dan_seq_atten_21_answer, dan_seq_atten_21_scores)

