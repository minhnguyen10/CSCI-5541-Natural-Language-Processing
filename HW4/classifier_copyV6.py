import argparse
from cmath import nan
# import imp
import os
import time
from unittest import result
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

    train_set = sentences[:int((len(sentences)+1)*.20)] #Remaining 80% to training set
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
        # clean_sentence = "<START> " + clean_sentence + "<END>"
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
def get_unigrams(corpus):
    vocab = set(words.words())
    unigrams={}
    for sentence in corpus:
        toks=word_tokenization(sentence)
        toks = ['<UNK>' if tok not in vocab else tok for tok in toks ]
        for k in toks:
            if k in unigrams:
                unigrams[k]+=1
            else:
                unigrams[k]=1
    return unigrams

def get_bigrams(corpus):
    vocab = set(words.words())
    # unigrams={}
    bigrams={}
    for sentence in corpus:
        tokens=word_tokenization(sentence)
        tokens = ['<UNK>' if tok not in vocab else tok for tok in tokens ]
        i=0
        # tokens = tokens.insert(0, "<START>")
        length=len(tokens)-1
        while i<length:
            if tokens[i] not in bigrams:
                bigrams[tokens[i]]={}
            if tokens[i+1] not in bigrams[tokens[i]]:
                bigrams[tokens[i]][tokens[i+1]]=1
            else:
                bigrams[tokens[i]][tokens[i+1]]+=1
            i+=1
    return bigrams


def getfrefre(bigram_model):
    frequencies = {}
    for w1 in bigram_model:
        for w2 in bigram_model[w1]:
            ngram_count = bigram_model[w1][w2]
            if ngram_count in frequencies:
                frequencies[ngram_count] += 1
            else:
                frequencies[ngram_count] = 1 
    return frequencies

def getfreferunigram(unigram_model):
    frequencies = {}
    for w1 in unigram_model:
        ngram_count = unigram_model[w1]
        if ngram_count in frequencies:
            frequencies[ngram_count] += 1
        else:
            frequencies[ngram_count] = 1 
    return frequencies


def get_trigrams(corpus):
	trigrams={}
	for sentence in corpus:
		tokens=word_tokenization(sentence)
		i=0
		length=len(tokens)-2
		while i<length:
			if tokens[i] not in trigrams:
				trigrams[tokens[i]]={}
			if tokens[i+1] not in trigrams[tokens[i]]:
				trigrams[tokens[i]][tokens[i+1]]={}
			if tokens[i+2] not in trigrams[tokens[i]][tokens[i+1]]:
				trigrams[tokens[i]][tokens[i+1]][tokens[i+2]]=1
			else:
				trigrams[tokens[i]][tokens[i+1]][tokens[i+2]]+=1
			i+=1
	return trigrams
    ## for all the word in vocabulary
    ## ngram assumption, store probability
    ## part in progress

def p_unigram(unigrams):
	uni_prob=[]
	total=0
	for w in unigrams:
		total+=unigrams[w]
	for w in unigrams:
		uni_prob.append([w,unigrams[w]/float(total)])
	return uni_prob

def p_bigram(unigrams,bigrams):
	bigram_prob=[]
 
	for w1 in bigrams:
		for w2 in bigrams[w1]:
			bigram_prob.append([w1,w2,bigrams[w1][w2]/float(unigrams[w1])])
	return bigram_prob

def get_bigram_toks(tokens):
    tok_bigram = {}
    i=0
    length=len(tokens)-1
    while i<length:
        if tokens[i] not in tok_bigram:
            tok_bigram[tokens[i]]={}
        if tokens[i+1] not in tok_bigram[tokens[i]]:
            tok_bigram[tokens[i]][tokens[i+1]]=1
        else:
            tok_bigram[tokens[i]][tokens[i+1]]+=1
        i+=1
    return tok_bigram

def smooth(c, k, train_vocab, frefrebigram):
    V = len(train_vocab)
    Ntable = sum(Counter(frefrebigram.values()))
    N0 = V*V - Ntable
    # print(frefrebigram.get(y))
    # Ncy1 = [0 if frefrebigram.get(y+1) == None else frefrebigram.get(y+1)]
    # Ncy = [0 if frefrebigram.get(y) == None else frefrebigram.get(y)]
    # smooth_c = 0
    if c < k:
        Ncy1 = N0 if frefrebigram.get(c+1) == None else frefrebigram.get(c+1)
        Ncy = N0 if frefrebigram.get(c) == None else frefrebigram.get(c)
        c = (c + 1) * Ncy1 / Ncy
    return c


# def findGoodTuringAdjustCount(c, k, nc_dict):
   
#     adjust_count = ( ( (( c + 1)*( nc_dict[c + 1] / nc_dict[c])) - ( c * (k+1) * nc_dict[k+1] / nc_dict[1]) ) /
#                      ( 1 - (( k + 1)*nc_dict[k + 1] / nc_dict[1]) )
#                    )
#     return adjust_count
    
def calculate_prob(unigram, bigram, frefrebigram, frefreunigram,dev_set):
    vocab = set(words.words())
    train_vocab = list(unigram.keys())
    probs = []
    for sentence in dev_set:
        toks = word_tokenization(sentence)
        # for tok in toks:
        #     if (tok not in vocab) or (tok not in train_vocab):
        #         tok = '<UNK>'
        #     else:
        #         tok = tok
        toks = ['<UNK>' if (tok not in vocab) or (tok not in train_vocab)  else tok for tok in toks]
        prob = 0
        bigram_sent = get_bigram_toks(toks)
        for w1 in bigram_sent:
            for w2 in bigram_sent[w1]:
                if (w1 not in bigram) or (w2 not in bigram[w1]):
                    nominator = 0
                else:
                    nominator = bigram[w1][w2]
                denominator = unigram[w1]
                
                # Apply GT smothing
                
                nominator = smooth(nominator, 1, train_vocab,frefrebigram)
                # print(nominator)
                # denominator = smooth(denominator, 5, train_vocab,frefreunigram)
                
                # This commentedl line is for Laplace
                # prob = prob  + np.log((nominator+1)/(denominator+len(train_vocab)))
                prob = prob  + np.log(nominator/denominator)
        # print(toks)
        probs.append(prob)
    return probs
    # for sentence in dev_set:


# Predict
def predict(unigrams, bigrams, frefrebigrams, frefreunigrams ,dev_set):
    # pred  = 
    results = []
    for i in range(len(bigrams)):
        probs = calculate_prob(unigrams[i], bigrams[i], frefrebigrams[i], frefreunigrams[i],dev_set)
        results.append(probs)
    results = np.vstack(results)    
    return results

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
        devtext = f.read()
        
        dev_sentences = sent_tokenize(devtext)
        dev_sentences_clean = sentence_cleaning(dev_sentences)
        
        train_sets = []
        author_name = []
        for file in files:
            author = file.replace(".txt", "")
            author = author.replace("_utf8", "")
            author_name.append(author)

            f = open(file)
            text = f.read()
            train_set = sent_tokenize(text)
            train_set_clean = sentence_cleaning(train_set)
            train_sets.append(train_set_clean)
        
        unigram_models = []
        bigram_models = []
        frefrebigrams = []
        frefreunigrams = []
        
        for i in range(len(train_sets)):
            # train the model, return the model
            # model = trigram(train_sets[i])
            # models.append(model)
            unigram = get_unigrams(train_sets[i])
            bigram = get_bigrams(train_sets[i])
            unigram_models.append(unigram)
            bigram_models.append(bigram)
            # print(bigram)
            frefrebigram = getfrefre(bigram)
            frefrebigrams.append(frefrebigram)
            frefreunigram = getfreferunigram(unigram)
            frefreunigrams.append(frefreunigram)
        
        print('DONE TRAINING!')   
        
        print("Results on a specific dev set:")
        dev_probs = predict(unigram_models,bigram_models, frefrebigrams, frefreunigrams,dev_sentences_clean)
        idx = np.argmax(dev_probs, axis = 0)
        result = Counter(idx)
        for i in range(len(author_name)):
            print(author_name[i] + " " + str(round((result[i]/len(dev_sentences_clean))*100, 2)) + "%") 



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


        # Building different n-gram model
        unigram_models = []
        bigram_models = []
        frefrebigrams = []
        frefreunigrams = []
        
        for i in range(len(train_sets)):
            # train the model, return the model
            # model = trigram(train_sets[i])
            # models.append(model)
            unigram = get_unigrams(train_sets[i])
            bigram = get_bigrams(train_sets[i])
            unigram_models.append(unigram)
            bigram_models.append(bigram)
            # print(bigram)
            frefrebigram = getfrefre(bigram)
            frefrebigrams.append(frefrebigram)
            frefreunigram = getfreferunigram(unigram)
            frefreunigrams.append(frefreunigram)
            


        print('DONE TRAINING!')   

        # Getting result from each model, print the highest probabilty as the lable
        print("Results on dev set:")
        
        # dev_sentence = ["This is a random sentence"]
        # print(calculate_prob(unigram_models[0], bigram_models[0], frefrebigrams[0],dev_sets[0]))
        for i in range(len(dev_sets)):
            # predict the development set using all training models
            # (we need to use all ngram models to find the one with highest prob)
            dev_probs = predict(unigram_models,bigram_models, frefrebigrams, frefreunigrams,dev_sets[i])
            # print(dev_probs)
            idx = np.argmax(dev_probs, axis = 0)
            pred_author = 0
            for k in idx:
                if k == i:
                    pred_author+=1
            # print result
            print(author_name[i] + "    " + str(pred_author) 
                  + " / " + str(len(dev_sets[i])) 
                  + " = " + str(round((pred_author/len(dev_sets[i]))*100, 2)) + "%"
                  + " correct")





if __name__ == "__main__":
    print("training... (this may take a while)")
    main()
