"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

from collections import defaultdict

def collect_word_stats(train):
    word_tag_counts = defaultdict(dict)
    tag_counts = {}
    for sentence in train:
        for word,tag in sentence:
            if word in word_tag_counts:
                if tag in word_tag_counts[word]:
                    word_tag_counts[word][tag] += 1
                else:
                    word_tag_counts[word][tag] = 1
            else:
                word_tag_counts[word][tag] = 1
            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1

    return word_tag_counts, tag_counts


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    word_tag_counts, tag_counts = collect_word_stats(train)
    word_most_freq_tag = {}
    result = [[] for _ in range(len(test))]

    for word,tags in word_tag_counts.items():
        max_tag = max(tags, key=tags.get)
        word_most_freq_tag[word] = max_tag

    most_freq_tag = max(tag_counts, key=tag_counts.get)
    print(most_freq_tag)

    for i in range(len(test)):
        for word in test[i]:
            result[i].append((word, word_most_freq_tag.get(word, most_freq_tag)))

    return result
