__author__ = 'yhd'

# import nltk
#
# print("... build")
# brown = nltk.corpus.brown
# corpus = [word.lower() for word in brown.words()]
#
# # Train on 95% f the corpus and test on the rest
# spl = int(95*len(corpus)/100)
# train = corpus[:spl]
# test = corpus[spl:]
#
# # Remove rare words from the corpus
# fdist = nltk.FreqDist(w for w in train)
# vocabulary = set(map(lambda x: x[0], filter(lambda x: x[1] >= 5, fdist.items())))
#
# train = map(lambda x: x if x in vocabulary else "*unknown*", train)
# test = map(lambda x: x if x in vocabulary else "*unknown*", test)
#
# print("... train")
# from nltk.model import NgramModel
# from nltk.probability import LidstoneProbDist
#
# estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
# lm = NgramModel(5, train, estimator=estimator)
#
# print("len(corpus) = %s, len(vocabulary) = %s, len(train) = %s, len(test) = %s" % ( len(corpus), len(vocabulary), len(train), len(test) ))
# print("perplexity(test) =", lm.perplexity(test))


# -*- coding: UTF-8-*-

import numpy
from math import log10

def prob_utterance(k,tab, l):
    '''
    :param k: 2**1__3
    :param tab: 1__3$2__4: count
    :param l: 1__3: [2]
    :return:
    '''

    '''calculates the probability of a sentence minus the end part given a sentence'''
    g = k.split()
    minus_end = 1
    for x in range(1,len(g)-1):
        word_split = g[x].split("**")
        word_split2 = g[x-1].split("**") #previous Gn
        trans_key =  word_split2[1]+'$'+word_split[1]
        if trans_key in tab.keys() and word_split2[1] in l.keys():
            trans_val = tab[trans_key] + 0.5
            num_previous_cat = len(l[word_split2[1]])+0.5
        else:
            trans_val = 0.5
            num_previous_cat = 0.5
        p_trans_temp = trans_val/num_previous_cat
        if word_split[1] in l.keys():
            p_emiss = (l[word_split[1]].count(word_split[0]) + 0.5)/(len(l[word_split[1]])+0.5)
        else:
            p_emiss = 0.5/0.5 #b/c the prob is gonna be 1/1 for every tiny category
        minus_end = minus_end * p_trans_temp * p_emiss
    return log10(minus_end)

def ends_of_utterance(j,tab,l):
    '''calculates the probability of the end part of the sentence'''
    g = j.split()
    ends = 1
    word_split = g[-1].split("**")
    word_split2 = g[-2].split("**") #previous Gn
    trans_key =  word_split2[1]+'$'+word_split[1]
    if trans_key in tab.keys():
        trans_val = tab[trans_key] + 0.5
    else:
        trans_val = 0.5
    if word_split2[1] in l.keys():
        num_previous_cat = len(l[word_split2[1]])+0.5
    else:
        num_previous_cat = 0.5
    ends = trans_val/num_previous_cat
    return log10(ends)

def calculate_perplexity(t_c,table,tab_list):
    '''calculates total probability of corpus using logs (and then taking them out)'''
    init_prob = 0
    test_perplex = 0
    for sent in t_c:
        sentence_prob = 0
        sentence_prob = prob_utterance(sent, table, tab_list) + ends_of_utterance(sent,table,tab_list)
        test_perplex += perplexity(sent, sentence_prob)
    corpus_perplexity = test_perplex/len(t_c)
    return corpus_perplexity

def perplexity(s, total_prob):
    '''calculates the perplexity of an utterance given the probability and sentence (for length)'''
    count = 0
    perplex = 0
    predone = 0
    count = len(s)-2 #not counting start and end
    predone = (-1/count)*total_prob
    perplex = numpy.log(numpy.power(10,predone))
    return perplex

import platform

def processor(test_sentences, train_source):
    for file_num in range(0,10,1):
        if platform.system() == 'Windows':
            infile1 = open(train_source,'r', encoding='utf-8')
        else:
            infile1 = open('data_root/train.txt','r')
        readinfile1 = infile1.readlines()

        # infile1.close()


        # readinfile1 = train_sentences
        readinfile2 = test_sentences


        test_cats = []

        ff_train_corpus = []
        for a_thing in readinfile1:
            if a_thing != '\n':
                new_a_string = "start! "+a_thing.rstrip()+" end!"
                ff_train_corpus.append(new_a_string.split())

        ff_test_corpus = []
        for c_thing in readinfile2:
            if c_thing != '\n':
                new_c_string = "start! "+c_thing.rstrip()+" end!"
                ff_test_corpus.append(new_c_string.split())
    #~*~*~*~*~*~*~*~*~*~* for FF
        ff_train_dict = {}
        new_ff_train = []
        for sentence in ff_train_corpus: #building the dictionary(ies) for training corpora
            new_ff_train_sentence = "start!**start!"
            for num in range(len(sentence)-2): #dont ignore start and end
                new_key = sentence[num]+"__"+sentence[num+2]  #DOUBLE UNDERSCORE!!!
                new_ff_train_sentence += " "+ sentence[num+1] + "**" + new_key +" "
                if new_key not in ff_train_dict.keys():
                    ff_train_dict[new_key] = [sentence[num+1]]
                else:
                    temp_list = ff_train_dict[new_key]
                    temp_list.append(sentence[num+1])
                    ff_train_dict[new_key] = temp_list
            new_ff_train_sentence += "end!**end!"
            new_ff_train.append(new_ff_train_sentence)


        new_ff_test = []
        for sentence in ff_test_corpus:
            new_ff_test_sentence = "start!**start!"
            for num in range(len(sentence)-2): #dont ignore start and end
                new_key = sentence[num]+"__"+sentence[num+2]  #DOUBLE UNDERSCORE!!!
                new_ff_test_sentence += " "+ sentence[num+1] + "**" + new_key +" "
            new_ff_test_sentence += "end!**end!"
            new_ff_test.append(new_ff_test_sentence)


        ff_word_list_train = []
        ff_word_list_test = []

        for sent in ff_test_corpus:
            for word_thing in sent:
                word_thing_s = word_thing.split("*")
                if word_thing_s[0] != "start!" and word_thing_s[0] != "end!":
                    ff_word_list_test.append(word_thing_s[0])

        ff_trans_cat_table = {}
        #preprocess!  turn into thing that looks like the english (and then do same thing to test)

        for sent in new_ff_train:
            sent_s = sent.split()
            for word_thing in sent_s:
                word_thing_s = word_thing.split("**")
                if word_thing_s[0] != "start!" and word_thing_s[0] != "end!":
                    ff_word_list_train.append(word_thing_s[0])
            for word_num in range(len(sent_s)-1):
                word = sent_s[word_num]
                both_word = word.split("**")
                next_word = sent_s[word_num+1]
                both_next_word = next_word.split("**")
                new_key = both_word[1]+'$'+both_next_word[1]
                if new_key not in ff_trans_cat_table.keys():
                    ff_trans_cat_table[new_key] = 1
                else:
                    temp = ff_trans_cat_table[new_key]
                    temp +=1
                    ff_trans_cat_table[new_key] = temp
                    #figure out something here for how to get frequency of "starts"

        ff_word_difference = [item for item in ff_word_list_test if item not in ff_word_list_train]
        ff_train_dict['glom'] = ff_word_difference

        #now making it frequent, needs to have .5% of types and .1% of tokens
        new_ff_train_dict = {}
        for thing in ff_train_dict.keys():
            if len(set(ff_train_dict[thing])) >= (.005)*len(set(ff_word_list_train)) and len(ff_train_dict[thing]) >= (.001)*len(ff_word_list_train) and len(set(ff_train_dict[thing])) > 2:
                new_ff_train_dict[thing] = ff_train_dict[thing]

        #print(len(new_ff_train_dict.keys()))

        #break

        ff_list = ff_train_dict.keys()

        return (new_ff_test, ff_trans_cat_table, ff_train_dict)


if __name__ == '__main__':
    # 1.01589408889
    s = ["I'm exhausted . __eou__ Okay , let's go home . __eou__"]

#     # 1.00799453797
#     ss = ["Can you manage chopsticks ? __eou__ Why not ? See . __eou__ Good mastery . How do you like our Chinese food ? __eou__ Oh , great ! It's delicious . You see , I am already putting on weight . There is one thing I don't like however , MSG . __eou__ What's wrong with MSG ? It helps to bring out the taste of the food . __eou__ According to some studies it may cause cancer . __eou__ Oh , don't let that worry you . If that were true , China wouldn't have such a large population . __eou__ I just happen to have a question for you guys . Why do the Chinese cook the vegetables ? You see what I mean is that most vitamin are destroyed when heated . __eou__ I don't know exactly . It's a tradition . Maybe it's for sanitary reasons . __eou__"]
#
#     # 1.00981532077

#     new_ff_test, ff_trans_cat_table, ff_train_dict = processor(
#         s,
#         sss
#     )
#
#     print('new_ff_test', new_ff_test)
#     print('ff_trans_cat_table', ff_trans_cat_table)
#     print('ff_train_dict', ff_train_dict)
#
#     results = calculate_perplexity(new_ff_test, ff_trans_cat_table, ff_train_dict)
#
#     print(results)