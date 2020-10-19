# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# def preprocess_train_set(train_set):
#     stop_words = {'ourselves', 'hers', 'between', 'yourself', 'again', 'there', 'about', 'once', 'during', 'out',
#                   'very',
#                   'having', 'with', 'they', 'own', 'an', 'be', 'some', 'do', 'its', 'yours', 'such', 'into', 'of',
#                   'most',
#                   'itself', 'other', 'off', 'is', 's', 'am', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves',
#                   'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this',
#                   'down',
#                   'should', 'our', 'their', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when',
#                   'at',
#                   'any', 'before', 'them', 'same', 'and', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what',
#                   'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just',
#                   'where',
#                   'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs',
#                   'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
#     new_set = [[] for i in range(len(train_set))]
#     for i in range(len(train_set)):
#         for word in train_set[i]:
#             if word not in stop_words:
#                 new_set[i].append(word)
#
#     return new_set


def get_u_word_stats(train_set, train_labels):
    pos_dict = {}
    neg_dict = {}
    for i in range(len(train_set)):
        cur_label = train_labels[i]
        for word in train_set[i]:
            if cur_label == 1:
                if word not in pos_dict:
                    pos_dict[word]=1
                else:
                    pos_dict[word]+=1
            else:
                if word not in neg_dict:
                    neg_dict[word]=1
                else:
                    neg_dict[word]+=1
    return pos_dict, neg_dict

def get_b_word_stats(train_set, train_labels):
    pos_dict = {}
    neg_dict = {}

    for i in range(len(train_set)):
        cur_label = train_labels[i]
        for j in range(len(train_set[i])-1):
            if cur_label == 1:
                if (train_set[i][j],train_set[i][j+1]) not in pos_dict:
                    pos_dict[(train_set[i][j],train_set[i][j+1])] = 1
                else:
                    pos_dict[(train_set[i][j],train_set[i][j+1])] += 1
            else:
                if (train_set[i][j],train_set[i][j+1]) not in neg_dict:
                    neg_dict[(train_set[i][j],train_set[i][j+1])] = 1
                else:
                    neg_dict[(train_set[i][j],train_set[i][j+1])] += 1
    return pos_dict, neg_dict


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=0.75, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set
    import math

    # get word positivity
    pos_dict, neg_dict = get_u_word_stats(train_set, train_labels)

    pos_word_count = sum(pos_dict.values())
    neg_word_count = sum(neg_dict.values())
    pos_vocab_size = len(pos_dict.keys())+1
    neg_vocab_size = len(neg_dict.keys())+1

    pos_denominator = pos_word_count+smoothing_parameter*pos_vocab_size
    neg_denominator = neg_word_count+smoothing_parameter*neg_vocab_size

    prediction = []
    for comment in dev_set:
        pos_chance = math.log(pos_prior)
        neg_chance = math.log(1-pos_prior)
        for word in comment:
            pos_chance += math.log((pos_dict.get(word, 0)+smoothing_parameter)/pos_denominator)
            neg_chance += math.log((neg_dict.get(word, 0)+smoothing_parameter)/neg_denominator)

        if pos_chance > neg_chance:
            prediction.append(1)
        else:
            prediction.append(0)

    return prediction

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=0.05, bigram_smoothing_parameter=0.01, bigram_lambda=0.5,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    import math

    # unigram parameters
    u_pos_dict, u_neg_dict = get_u_word_stats(train_set, train_labels)

    pos_word_count = sum(u_pos_dict.values())
    neg_word_count = sum(u_neg_dict.values())
    u_pos_vocab_size = len(u_pos_dict.keys()) + 1
    u_neg_vocab_size = len(u_neg_dict.keys()) + 1

    pos_denominator = pos_word_count + unigram_smoothing_parameter * u_pos_vocab_size
    neg_denominator = neg_word_count + unigram_smoothing_parameter * u_neg_vocab_size

    # bigram parameters
    b_pos_dict, b_neg_dict = get_b_word_stats(train_set, train_labels)

    pos_pair_count = sum(b_pos_dict.values())
    neg_pair_count = sum(b_neg_dict.values())
    b_pos_vocab_size = len(b_pos_dict.keys()) + 1
    b_neg_vocab_size = len(b_neg_dict.keys()) + 1

    pos_pair_denominator = pos_pair_count + bigram_smoothing_parameter * b_pos_vocab_size
    neg_pair_denominator = neg_pair_count + bigram_smoothing_parameter * b_neg_vocab_size

    prediction = []
    for comment in dev_set:
        u_pos_chance = math.log(pos_prior)
        u_neg_chance = math.log(1 - pos_prior)

        b_pos_chance = math.log(pos_prior)
        b_neg_chance = math.log(1 - pos_prior)

        for i in range(len(comment)-1):
            u_pos_chance += math.log((u_pos_dict.get(comment[i], 0) + unigram_smoothing_parameter) / pos_denominator)
            u_neg_chance += math.log((u_neg_dict.get(comment[i], 0) + unigram_smoothing_parameter) / neg_denominator)

            b_pos_chance += math.log((b_pos_dict.get((comment[i], comment[i+1]), 0) + bigram_smoothing_parameter) / pos_pair_denominator)
            b_neg_chance += math.log((b_neg_dict.get((comment[i], comment[i+1]), 0) + bigram_smoothing_parameter) / neg_pair_denominator)

        u_pos_chance += math.log((u_pos_dict.get(comment[-1], 0) + unigram_smoothing_parameter) / pos_denominator)
        u_neg_chance += math.log((u_neg_dict.get(comment[-1], 0) + unigram_smoothing_parameter) / neg_denominator)

        hybrid_pos_chance = (1 - bigram_lambda) * u_pos_chance + bigram_lambda * b_pos_chance
        hybrid_neg_chance = (1 - bigram_lambda) * u_neg_chance + bigram_lambda * b_neg_chance

        if hybrid_pos_chance > hybrid_neg_chance:
            prediction.append(1)
        else:
            prediction.append(0)

    return prediction
