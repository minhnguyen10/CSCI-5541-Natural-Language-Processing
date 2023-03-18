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
        # clean_sentence = "<START> " + clean_sentence + "<END>"
        clean_sentences.append(clean_sentence)

    return clean_sentences


def class_UNK(tok):
    if tok.endswith('ive'):
        unktok = '<UNKIVE>'
    elif tok.endswith('able'):
        unktok = '<UNKABLE>'
    elif tok.endswith('ed'):
        unktok = '<UNKED>'
    elif tok.endswith('ing'):
        unktok = '<UNKING>'
    else:
        unktok = '<UNK>'
    return unktok

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
        toks = word_tokenization(sentence)
        toks = ['<UNK>' if tok not in vocab else tok for tok in toks ]
        for k in toks:
            if k in unigrams:
                unigrams[k] += 1
            else:
                unigrams[k] = 1
    return unigrams

def get_bigrams(corpus):
    vocab = set(words.words())
    bigram={}
    for sentence in corpus:
        toks = word_tokenization(sentence)
        toks = ['<UNK>' if tok not in vocab else tok for tok in toks ]
        i=0
        # tokens = tokens.insert(0, "<START>")
        # length = len(toks) - 1
        while i < (len(toks) - 1):
            if toks[i] not in bigram:
                bigram[toks[i]] = {}
            if toks[i+1] not in bigram[toks[i]]:
                bigram[toks[i]][toks[i+1]] = 1
            else:
                bigram[toks[i]][toks[i+1]] += 1
            i += 1
    return bigram

def get_bigram_toks(toks):
    tok_bigram = {}
    i = 0
    # length=len(tokens)-1
    while i < (len(toks) - 1):
        if toks[i] not in tok_bigram:
            tok_bigram[toks[i]]={}
        if toks[i+1] not in tok_bigram[toks[i]]:
            tok_bigram[toks[i]][toks[i+1]] = 1
        else:
            tok_bigram[toks[i]][toks[i+1]] += 1
        i += 1
    return tok_bigram


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

def getfrefretrigram(trigram_model):
    frequencies = {}
    for w1 in trigram_model:
        for w2 in trigram_model[w1]:
            for w3 in trigram_model[w1][w2]:
                ngram_count = trigram_model[w1][w2][w3]
                if ngram_count in frequencies:
                    frequencies[ngram_count] += 1
                else:
                    frequencies[ngram_count] = 1 
    return frequencies


def get_trigrams(corpus):
	trigrams={}
	for sentence in corpus:
		toks=word_tokenization(sentence)
		i = 0
		# length =len(toks)-2
		while i < (len(toks)-2):
			if toks[i] not in trigrams:
				trigrams[toks[i]]={}
			if toks[i+1] not in trigrams[toks[i]]:
				trigrams[toks[i]][toks[i+1]] = {}
			if toks[i+2] not in trigrams[toks[i]][toks[i+1]]:
				trigrams[toks[i]][toks[i+1]][toks[i+2]] = 1
			else:
				trigrams[toks[i]][toks[i+1]][toks[i+2]] += 1
			i += 1
	return trigrams

def get_trigrams_toks(toks):
	trigrams={}
	i = 0
	# length=len(toks)-2
	while i < (len(toks)-2):
		if toks[i] not in trigrams:
			trigrams[toks[i]]={}
		if toks[i+1] not in trigrams[toks[i]]:
			trigrams[toks[i]][toks[i+1]] = {}
		if toks[i+2] not in trigrams[toks[i]][toks[i+1]]:
			trigrams[toks[i]][toks[i+1]][toks[i+2]] = 1
		else:
			trigrams[toks[i]][toks[i+1]][toks[i+2]] += 1
		i += 1
	return trigrams



def smooth(c, k, train_vocab, frefre):
    V = len(train_vocab)
    Ntable = sum(Counter(frefre.values()))
    N0 = V*V - Ntable
    # print(frefrebigram.get(y))
    # Ncy1 = [0 if frefrebigram.get(y+1) == None else frefrebigram.get(y+1)]
    # Ncy = [0 if frefrebigram.get(y) == None else frefrebigram.get(y)]
    # smooth_c = 0
    if c < k:
        Ncy1 = N0 if frefre.get(c+1) == None else frefre.get(c+1)
        Ncy = N0 if frefre.get(c) == None else frefre.get(c)
        c = (c + 1) * Ncy1 / Ncy
    return c

def smooth3(c, k, train_vocab, frefre):
    V = len(train_vocab)
    Ntable = sum(Counter(frefre.values()))
    N0 = V**3 - Ntable
    # print(frefrebigram.get(y))
    # Ncy1 = [0 if frefrebigram.get(y+1) == None else frefrebigram.get(y+1)]
    # Ncy = [0 if frefrebigram.get(y) == None else frefrebigram.get(y)]
    # smooth_c = 0
    if c < k:
        Ncy1 = N0 if frefre.get(c+1) == None else frefre.get(c+1)
        Ncy = N0 if frefre.get(c) == None else frefre.get(c)
        c = (c + 1) * Ncy1 / Ncy
    return c
    
def calculate_prob(unigram, bigram, trigram,frefrebigram, frefretrigram,dev_set):
    vocab = set(words.words())
    train_vocab = list(unigram.keys())
    probs = []
    for sentence in dev_set:
        toks = word_tokenization(sentence)
        toks = ['<UNK>' if (tok not in vocab) or (tok not in train_vocab)  else tok for tok in toks]
        prob = 0
        trigram_sent = get_trigrams_toks(toks)
        # new_prob = 0
        for w1 in trigram_sent:
            for w2 in trigram_sent[w1]:
                for w3 in trigram_sent[w1][w2]:
                    new_prob = 0
                    # if (w1 not in trigram)
                    if (w1 in trigram) and (w2 in trigram[w1]) and (w3 in trigram[w1][w2]):
                        nominator = trigram[w1][w2][w3]
                        denominator = bigram[w1][w2]
                        # nominator = smooth3(nominator, 6, train_vocab, frefretrigram)
                        # denominator =smooth(denominator, 6, train_vocab, frefrebigram)
                        new_prob = nominator/denominator
                    elif (w1 in trigram) and (w2 in trigram[w1]) and (w3 not in trigram[w1][w2]):
                        nominator = bigram[w1][w2]
                        denominator = unigram[w1]
                        # nominator = smooth(nominator, 6, train_vocab, frefretrigram)
                        new_prob = 0.001*(nominator/denominator)
                    else:
                        new_prob = 0.001*0.0001*unigram[w1]

                    # Apply GT smothing
                
                    # This commentedl line is for Laplace
                    # prob = prob  + np.log((nominator+1)/(denominator+len(train_vocab)))
                    prob = prob  + np.log(new_prob)
        # print(toks)
        probs.append(prob)
    return probs
    # for sentence in dev_set:


# Predict
def predict(unigrams, bigrams, trigrams,frefrebigrams, frefretrigrams ,dev_set):
    # pred  = 
    results = []
    for i in range(len(bigrams)):
        probs = calculate_prob(unigrams[i], bigrams[i], trigrams[i],frefrebigrams[i], frefretrigrams[i],dev_set)
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
        trigram_models = []
        frefrebigrams = []
        frefretrigrams = []
        
        for i in range(len(train_sets)):
            print("Load " + author_name[i] +" ...")
            # train the model, return the model
            # model = trigram(train_sets[i])
            # models.append(model)
            unigram = get_unigrams(train_sets[i])
            bigram = get_bigrams(train_sets[i])
            unigram_models.append(unigram)
            bigram_models.append(bigram)
            trigram =  get_trigrams(train_sets[i])
            trigram_models.append(trigram)
            # print(bigram)
            frefrebigram = getfrefre(bigram)
            frefrebigrams.append(frefrebigram)
            frefretrigram = getfrefretrigram(trigram)
            frefretrigrams.append(frefretrigram)
        
        print('DONE TRAINING!')   
        
        print("Results on a specific dev set:")
        dev_probs = predict(unigram_models,bigram_models, trigram_models,frefrebigrams, frefretrigrams,dev_sentences_clean)
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
        trigram_models = []
        frefrebigrams = []
        frefretrigrams = []
        
        for i in range(len(train_sets)):
            # train the model, return the model
            # model = trigram(train_sets[i])
            # models.append(model)
            unigram = get_unigrams(train_sets[i])
            bigram = get_bigrams(train_sets[i])
            trigram =  get_trigrams(train_sets[i])
            unigram_models.append(unigram)
            bigram_models.append(bigram)
            trigram_models.append(trigram)
            # print(bigram)
            frefrebigram = getfrefre(bigram)
            frefrebigrams.append(frefrebigram)
            frefretrigram = getfrefretrigram(trigram)
            frefretrigrams.append(frefretrigram)
            


        print('DONE TRAINING!')   

        # Getting result from each model, print the highest probabilty as the lable
        print("Results on dev set:")
        
        # dev_sentence = ["This is a random sentence"]
        # print(calculate_prob(unigram_models[0], bigram_models[0], frefrebigrams[0],dev_sets[0]))
        for i in range(len(dev_sets)):
            # predict the development set using all training models
            # (we need to use all ngram models to find the one with highest prob)
            dev_probs = predict(unigram_models,bigram_models, trigram_models,frefrebigrams, frefretrigrams,dev_sets[i])
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
