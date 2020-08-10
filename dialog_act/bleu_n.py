
from nltk.translate.bleu_score import corpus_bleu

# reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
# candidate = ['this', 'is', 'a', 'test']

reference = []
candidate = []

import codecs

# reference_txt = 'metrics/sphred_4_answer.txt'
# candidate_txt = 'metrics/sphred_4_reply.txt'

# reference_txt = 'metrics/seq_torch_batch_answer.txt'
# candidate_txt = 'metrics/seq_torch_batch_reply.txt'

reference_txt = 'metrics/hred_dan_answer.txt'
candidate_txt = 'metrics/hred_dan_reply.txt'


# reference_txt = 'models_encoder_decoder/encoder_decoder_answer.txt'
# candidate_txt = 'models_encoder_decoder/encoder_decoder_reply.txt'


with codecs.open(reference_txt, mode="r", encoding='utf-8') as f:
    for line in f.readlines():
        reference.append([line.rstrip('\n').split(' ')])


with codecs.open(candidate_txt, mode="r", encoding='utf-8') as f:
    for line in f.readlines():
        candidate.append(line.rstrip('\n').split(' '))


score1 = corpus_bleu(reference, candidate, weights=(1, 0, 0, 0))
score2 = corpus_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
score3 = corpus_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
score4 = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

print(score1)
print(score2)
print(score3)
print(score4)


# seq2seq

# 0.074517400474
# 0.0121118481653
# 0.00347947589713
# 0.00119814267947

# seq2seq + fae

# 0.124517050839
# 0.0316446350548
# 0.0122962610775
# 0.00491658135603

# attention

# 0.0872398628036
# 0.0145609924435
# 0.00436628012055
# 0.00160532096091

# attention + fae

# 0.0884193372547
# 0.0183554611673
# 0.00598478623112
# 0.00197020908547

# hred

# 0.1033094367
# 0.0242481805506
# 0.00871641705649
# 0.0032944828112

# hred + fae




# sphred

# 0.0871818884772
# 0.0199748776515
# 0.00750149088857
# 0.0029324008619

# seqGAN

# 0.099439371906
# 0.0221708830731
# 0.00794662793889
# 0.00301922218835


# not multi-task learning

# 0.114019289008
# 0.0183319054272
# 0.00465260261567
# 0.00118704237193