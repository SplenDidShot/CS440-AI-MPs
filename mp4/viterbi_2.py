"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""
from collections import defaultdict
import math

def collect_init_trans_emission_stats(train):
    init_tag_counts = {}
    tag_pair_counts = defaultdict(dict)
    tag_word_counts = defaultdict(dict)
    for sentence in train:
        sentence = sentence[1:-1]
        if sentence[0][1] not in init_tag_counts:
            init_tag_counts[sentence[0][1]] = 1
        else:
            init_tag_counts[sentence[0][1]] += 1

        for i in range(len(sentence)-1):
            word0, tag0 = sentence[i]
            word1, tag1 = sentence[i+1]
            if tag0 not in tag_pair_counts or tag1 not in tag_pair_counts[tag0]:
                tag_pair_counts[tag0][tag1] = 1
            else:
                tag_pair_counts[tag0][tag1] += 1

            if tag0 not in tag_word_counts or word0 not in tag_word_counts[tag0]:
                tag_word_counts[tag0][word0] = 1
            else:
                tag_word_counts[tag0][word0] += 1

        if tag1 not in tag_word_counts or word1 not in tag_word_counts[tag1]:
            tag_word_counts[tag1][word1] = 1
        else:
            if word1 in tag_word_counts[tag1]:
                tag_word_counts[tag1][word1] += 1

    # tag to tag_count
    hapax_words = {}
    for tag, word_dict in tag_word_counts.items():
        for word, word_count in word_dict.items():
            if word_count == 1:
                if tag not in hapax_words:
                    hapax_words[tag] = 1
                else:
                    hapax_words[tag] += 1

    return init_tag_counts, tag_pair_counts, tag_word_counts, hapax_words

def compute_init_transition_emission_probabilities(init_tag_probs, tag_pair_probs, tag_word_probs, hapax_probs,
                                                   smoothing_param):
    # compute initial tag probabilities
    total_tag_counts = sum(init_tag_probs.values())
    tag_types = len(init_tag_probs.keys()) + 1
    denominator = total_tag_counts + smoothing_param * tag_types

    for tag in init_tag_probs.keys():
        init_tag_probs[tag] = math.log((init_tag_probs[tag] + smoothing_param) / denominator)
    init_unseen_prob = math.log(smoothing_param / denominator)

    # compute transitional probabilities
    trans_unseen_prob = {}
    for tag0 in tag_pair_probs.keys():
        next_tag_total_count = sum(tag_pair_probs[tag0].values())
        next_tag_types = len(tag_pair_probs[tag0].keys()) + 1
        denominator = next_tag_total_count + smoothing_param * next_tag_types

        for tag1 in tag_pair_probs[tag0].keys():
            tag_pair_probs[tag0][tag1] = math.log((tag_pair_probs[tag0][tag1] + smoothing_param) / denominator)
        trans_unseen_prob[tag0] = math.log(smoothing_param / denominator)
    trans_unseen_prob['END'] = math.log(smoothing_param / denominator)

    # compute probabilities for tags with hapax words
    total_hapax_count = sum(hapax_probs.values())
    total_hapax_tags = len(hapax_probs.keys()) + 1
    denominator = total_hapax_count + smoothing_param * total_hapax_tags
    for tag, tag_count in hapax_probs.items():
        hapax_probs[tag] = math.log((tag_count + smoothing_param)/denominator)
    hapax_unseen = math.log(smoothing_param/denominator)

    # compute emission probabilities
    emission_unseen_prob = {}
    for tag in tag_word_probs.keys():
        total_words_under_tag_count = sum(tag_word_probs[tag].values())
        vocab_size_under_tag = len(tag_word_probs[tag]) + 1
        emission_smoothing_param = smoothing_param
        denominator = total_words_under_tag_count + emission_smoothing_param * vocab_size_under_tag
        for word in tag_word_probs[tag].keys():
            tag_word_probs[tag][word] = math.log((tag_word_probs[tag][word] + emission_smoothing_param) / denominator)
        emission_unseen_prob[tag] = math.log(emission_smoothing_param / denominator)

    return init_tag_probs, tag_pair_probs, tag_word_probs, trans_unseen_prob, emission_unseen_prob, hapax_probs, \
           hapax_unseen

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    smoothing_param = 0.00001
    init_tag_counts, tag_pair_counts, tag_word_counts, hapax_words = collect_init_trans_emission_stats(train)
    init_tag_probs, transmission_probs, emission_probs,trans_unseen_prob, emission_unseen_prob, hapax_probs, hapax_unseen = \
        compute_init_transition_emission_probabilities(init_tag_counts,
                                                       tag_pair_counts,
                                                       tag_word_counts,
                                                       hapax_words,
                                                       smoothing_param)
    tag_result = []
    for sentence in test:
        trimmed_sentence = sentence[1:-1]
        v = [{}]
        for tag in init_tag_probs.keys():
            v[0][tag] = {
                'p': init_tag_probs[tag] + emission_probs[tag].get(trimmed_sentence[0],
                       emission_unseen_prob[tag] + hapax_probs.get(tag,hapax_unseen)),
                'b': None}

        for i in range(1, len(trimmed_sentence)):
            v.append({})
            for curr_tag in emission_probs.keys():
                max_trans_prob = -999999
                back_tag = None
                for prev_tag in v[i - 1].keys():
                    curr_trans_prob = v[i - 1][prev_tag]['p'] + transmission_probs[prev_tag].get(curr_tag,
                                                                                                 trans_unseen_prob[
                                                                                                     prev_tag])
                    if curr_trans_prob > max_trans_prob:
                        max_trans_prob = curr_trans_prob
                        back_tag = prev_tag
                max_prob = max_trans_prob + emission_probs[curr_tag].get(trimmed_sentence[i],
                                                                         emission_unseen_prob[curr_tag] +
                                                                         hapax_probs.get(curr_tag,hapax_unseen))
                v[i][curr_tag] = {'p': max_prob, 'b': back_tag}

        best_path = [(sentence[-1], 'END')]

        max_prob = -99999
        last_tag = None
        for tag, pb_dict in v[-1].items():
            if pb_dict['p'] > max_prob:
                max_prob = pb_dict['p']
                last_tag = tag
        best_path.append((trimmed_sentence[-1], last_tag))
        for i in range(len(trimmed_sentence) - 2, -1, -1):
            best_path.append((trimmed_sentence[i], v[i + 1][last_tag]['b']))
            last_tag = v[i + 1][last_tag]['b']
        best_path.append((sentence[0], 'START'))
        tag_result.append(best_path[::-1])

    return tag_result