import argparse
from cmath import nan
# import imp
import os
import time
from bleach import clean
import numpy as np
import sys
import re
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
import nltk
from nltk.corpus import words
from collections import Counter
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import random
# from gensim import summarization
# from spacy.lang.en import English
# summarization.textcleaner.clean_text_by_word

# Data Splitting
# Given a list of all senetences
def splitdata(sentences):
    random.seed(10)
    random.shuffle(sentences)

    train_set = sentences[:int((len(sentences)+1)*.80)] #Remaining 80% to training set
    dev_set = sentences[int((len(sentences)+1)*.80):] #Splits 20% data to test set
    return train_set, dev_set

# Text cleaning
def sentence_cleaning(sentences):
    clean_sentences = []
    for sentence in sentences:
        clean_sentence = sentence.replace('\n','')
        clean_sentence = sentence.replace('\"','')
        clean_sentence = sentence.replace('—',' ')
        clean_sentence = sentence.replace(',',' ')
        clean_sentence = re.sub("\s+"," ", clean_sentence) # remove extra blank
        clean_sentence = re.sub(".$","", clean_sentence)    # remove
        clean_sentence = re.sub("[^-9A-Za-z ]", "" , clean_sentence)

        clean_sentences.append(clean_sentence)

    return clean_sentences

# Word tokenization
def word_tokenization(sentence):
    tokens = word_tokenize(sentence)
    wn = WordNetLemmatizer()
    words = [wn.lemmatize(token) for token in tokens]
    tok = [word.lower() for word in words]
    return tok


# Models
def ngram (train_set, n = 2):
    ngram = []
    for sentence in train_set:
        toks = word_tokenization(sentence)
        toks = ['_UNK_' if tok not in words.words() else tok for tok in toks ]
        ngram = [toks[i] + ' ' + toks[i+1] for i in range(len(toks) - n +1)]
        ngramset.extend(ngrams)
    ngramdict = Counter(ngramset)  ## dictionary stores frequencies of all ngram

    ## for all the word in vocabulary
    ## ngram assumption, store probability
    ## part in progress

    for key in ngramdict:
        w = key.split(' ')[-1]



    return model


# Predict
def predict(model, dev_set):
    correct = nan
    return correct


# Main class that read option
def main():
    ## initialize parser
    parser = argparse.ArgumentParser()
    ## add argument
    parser.add_argument("input")
    parser.add_argument( "-test", help = "output filename")
    args = parser.parse_args()

    f = open(args.input)
    files = []
    for line in f:
        files.append(line.strip())


    # if test option is chosen
    if args.test:
    # print(args.test)
        f = open(args.test)
        text = f.read()

        for line in text:
                # tokens = sent_tokenize(line)
            # print(summarization.textcleaner.clean_text_by_word(line))
            print("")


    else:
        # split the data into test and training set
        train_sets = []
        dev_sets = []
        author_name = []

        for file in files:
            # get author name
            author = file.replace(".txt", "")
            author = author.replace("_utf8", "")
            author_name.append(author)

            # Sentence tokenization
            f = open(file)
            text = f.read()
            sentences = sent_tokenize(text)
            train_set, dev_set = splitdata(sentences)
            train_set_clean = sentence_cleaning(train_set)
            dev_set_clean = sentence_cleaning(dev_set)
            train_sets.append(train_set_clean)
            dev_sets.append(dev_set_clean)
            # print(len(train_set))
            # print(dev_set_clean)

            # you can use this word_tokenization method for the model
            # for example
            for sentence in dev_set_clean:
                print(word_tokenization(sentence))

            # print(len(sentences))
        # print(len(train_sets))
        # print(dev_sets)


        # Building different n-gram model
        models = []
        for i in range(len(train_sets)):
            # train the model, return the model
            model = trigram(train_sets[i])
            models.append(model)

        # Getting result from each model, print the highest probabilty as the lable
        print("Results on dev set:")
        for i in range(len(dev_sets)):
            # predict the development set using all training models
            # (we need to use all ngram models to find the one with highest prob)
            dev_result = predict(models, dev_set)

            # print result
            print(author_name[i] + "    " + str(dev_result) + " / " + str(len(dev_set)) + " correct")





if __name__ == "__main__":
    print("training... (this may take a while)")
    main()
