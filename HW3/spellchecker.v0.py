import sys
import re

import numpy as np
from torch import threshold

def tokenize2(fname):
    '''Incomplete adaptation of the book's tokenization
    algorithm in figure 3.22
    Parameter:
        a string containing the name of a file
        to read from
    Return Value:
        a list of strings containing individual tokens
    '''
    abbr = ["Co.", "Corp.", "vs.", "e.g.", "etc.", "ex.", "cf.",
            "eg.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.",
            "Sept.", "Oct.", "Nov.", "Dec.", "jan.", "feb.", "mar.",
            "apr.", "jun.", "jul.", "aug.", "sept.", "oct.", "nov.",
            "dec.", "ed.", "eds.", "repr.", "trans.", "vol.", "vols.",
            "rev.", "est.", "b.", "m.", "bur.", "d.", "r.", "M.", "Dept.",
            "MM.", "U.", "Mr.", "Jr.", "Ms.", "Mme.", "Mrs.", "Dr.",
            "Ph.D."]
    with open(fname) as fp:
        tokens = []
        new_toks = []
        for line in fp:
            # unambiguous separators
            line = re.sub(r'([\\?!()\";/\\|`])', r' \1 ', line)
            
            # whitespace around commas that aren't in numbers
            line = re.sub(r'([^0-9]),', r'\1 , ', line)
            line = re.sub(r',([^0-9])', r' , \1', line)
            
            # distinguish singlequotes from apostrophes by
            # segmenting off singke quotes not precede by letter
            line = re.sub(r"^(')", r'\1 ', line) #not sure about this
            line = re.sub(r"([^A-Za-z0-9])'", r"\1 '", line)
            
            # segment off unambiguous word-final clitics and punctuation 
            line = re.sub(r"('|:|-|'S|'D|'M|'LL|'RE|'VE|N'T|'s|'d|'m|'ll|'re|'ve|n't)", r' \1', line)
            line = re.sub(r'(\'|:|-|\'S|\'D|\'M|\'LL|\'RE|\'VE|N\'T|\'s|\'d|\'m|\'ll|\'re|\'ve|n\'t)([^A-Za-z0-9])', 
                          r' \1 \2', line) #not sure
            
            
            #  now deal with period
            tokens.extend(line.split())
            pattern_1 = r".*[a-zA-Z0-9]\."
            pattern_2 = r"^([A-Za-z]\.([A-Za-z]\.)+|[A-Z][bcdfghj-nptvxz]+\.)$"
        for word in tokens:
            if (re.match(pattern_1, word)) and not (re.match(pattern_2, word)) and (word not in abbr):
                    #new_toks.append(word[:-1])
                    #new_toks.append(".")
                wordzz = re.sub(r"(.*[a-zA-Z0-9])(\.)$", r"\1", word)
                new_toks.append(wordzz) 
                new_toks.append(".") 
            else:
                new_toks.append(word)                      
                    
    #print(tokens)
    print(new_toks)
    return new_toks


# def printDistances(distances, token1Length, token2Length):
#     for t1 in range(token1Length + 1):
#         for t2 in range(token2Length + 1):
#             print(int(distances[t1][t2]), end=" ")
#         print()
        
def edit_dist(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1] + 1
                b = distances[t1 - 1][t2] + 1
                c = distances[t1 - 1][t2 - 1] + 2
                distances[t1][t2] = min(a,b,c)

    # printDistances(distances, len(token1), len(token2))
    return distances[len(token1)][len(token2)]


# def edit_dist:
def suggest(token):
    suggest_list = []
    correct_words = []
    words_dict = {}
    with open('words') as f:
        lines = f.readlines()
        for line in lines:
            for word in line.split():         
                correct_words.append(word)
    # print(suggest_list)
    for word in correct_words:
        words_dict[word] = edit_dist(word,token)     
    threshold_list = 3
    suggest_list = sorted(words_dict, key=words_dict.get, reverse=False)[:threshold_list]
    return suggest_list    
    
#     return 
    
def main():
    # f = open('text.txt',"r")
    #f = open(sys.argv[1])
    # for word in best_words(f, topwords):
    #     print(word)
    dict = tokenize2(sys.argv[1])
    distance = edit_dist("kelm", "hello")
    print(distance)
    print(suggest("helloa"))
    #print(dict)
    #f.close()

if __name__ == '__main__':
   main()