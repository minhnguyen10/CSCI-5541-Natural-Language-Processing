from pydoc_data.topics import topics
import nltk
import string
from collections import Counter
import pickle
from nltk.corpus import words
# nltk.download('brown')
from nltk import WordNetLemmatizer
import argparse
from collections import defaultdict

from pandas import concat


# Buidling model by counting occurance of topics and words
def model_build(topics_list,words_list):
    topics_list = ['<START>'] + topics_list + ['<STOP>']
    words_list = ['<START>'] + words_list + ['<STOP>']
    all_bigram_tags = [tuple([topics_list[i],topics_list[i+1]]) for i in range(len(topics_list)-1)]
    ugram_tag = Counter(topics_list)
    bgram_tag = Counter(all_bigram_tags)
    ugram_tw = Counter(zip(topics_list, words_list))

    return ugram_tag, bgram_tag, ugram_tw

def calc_prob(ugram_tag, bgram_tag, ugram_tw):
    transition_mat = {} ## A matrix
    emission_mat = {} ## B matrix
    
    ### construct transition matrix
    for key in bgram_tag:
        t_1, t_2 = key
        
        if t_1 not in transition_mat:
            transition_mat[t_1] = {}
        transition_mat[t_1][t_2] = bgram_tag[key]/ugram_tag[t_1]
    
    ### construct emission matrix
    for key in ugram_tw:
        t, w = key
        if t not in emission_mat:
            emission_mat[t] = {}
        emission_mat[t][w] = ugram_tw[key]/ugram_tag[t]
    return transition_mat, emission_mat


def get_stationary(ugram_tag,topics_list):
    stationary = {}
    total = len(topics_list)
    for t in ugram_tag:
        stationary[t] = ugram_tag[t]/total
    
    return stationary

def main():
    print("building hmm model...")
    topics_list = []
    words_list = []
    parser = argparse.ArgumentParser()
    ## add argument
    parser.add_argument("text")
    args = parser.parse_args()
    file1 = open(args.text, 'r')
    Lines = file1.readlines()
    
    count = 0
    # Strips the newline character
    for line in Lines:
        topics_list.append(line.split()[0])
        words_list.append(line.split()[1])
        # print(words_list)
    ugram_tag, bgram_tag, ugram_tw = model_build(topics_list,words_list)
    transition_mat, emission_mat = calc_prob(ugram_tag, bgram_tag, ugram_tw)
    
    # model = [transition_mat, emission_mat]
    # stationary = get_stationary(ugram_tag,topics_list)


    print("Answering Questions:")
    #Question
    print("a.")
    print('transition probability P(baseball|religion) is {}'.format(transition_mat['religion']['baseball']))
    print("b.")
    print('transition probability P(windows|windows) is {}'.format(transition_mat['windows']['windows']))
    print("c.")
    print('emission probability P(god|medicine) is {}'.format(emission_mat['medicine']['god']))
    print("d.")
    print('emission probability P(spirit|baseball) is {}'.format(emission_mat['baseball']['spirit']))
    print("e.")
    print("- calculating using equation and example from page 180-181 in the textbook")
    
    
    
    # prob1 = transition_mat['windows']['windows']*emission_mat['windows']['com']*emission_mat['windows']['re']
    # print('P(windows windows,com re) is {}'.format(prob1))
    

    states = list(transition_mat.keys())
    states.remove('<START>')
    # states = states.remove('<STOP>')
    # Since i am not so sure if descending order specific to (e), I just assume that is true
    probs = {}
    for start_state in states:
        for end_state in states:
            prob1 = transition_mat[start_state][end_state]*emission_mat[start_state]['com']*emission_mat[end_state]['re']
            # print('P(',start_state,'',end_state,') is', str(prob1))
            key = start_state+' '+end_state
            probs[key] = prob1
    sorted_prop = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for event in sorted_prop:
        print("P(",event[0],",com re ) is", str(event[1]))

    
if __name__ == "__main__" :
    main()