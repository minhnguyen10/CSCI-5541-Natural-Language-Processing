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
def get_unigrams(corpus):
	unigrams={}
	for sentence in corpus:
		tokens=word_tokenization(sentence)
		for k in tokens:
			if k in unigrams:
				unigrams[k]+=1
			else:
				unigrams[k]=1
	return unigrams

def get_bigrams(corpus):
	bigrams={}
	for sentence in corpus:
		tokens=word_tokenization(sentence)
		i=0
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

# Predict
def predict(models, dev_sentence):
    # pred  = 
    dev_sentence = [dev_sentence]
    dev_sentence_bigram = p_unigram(dev_sentence)
    Prob_bigram = []
    for i in range(len(models)):
        Prob = p_bigram
    return pred


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
            # model = trigram(train_sets[i])
            # models.append(model)
            unigram = get_unigrams(train_sets[i])
            bigram = get_bigrams(train_sets[i])
            bigram_prop = p_bigram(unigram,bigram)
            models.append(bigram_prop)
            
        
        # for i in range(len(dev_sets)):
        #     accuracy = 0    
        #     for sentence in dev_sets[i]:
        #         pred = predict(models, sentence)
        #         if pred == i:
        #             accuracy += 1
        #     print(accuracy)
            

        # Getting result from each model, print the highest probabilty as the lable
        print("Results on dev set:")
        # dev_sentence = ["This is a random sentence"]
        # print(get_bigrams(dev_sentence))
        # for i in range(len(dev_sets)):
        #     # predict the development set using all training models
        #     # (we need to use all ngram models to find the one with highest prob)
        #     dev_result = predict(models, dev_set)

        #     # print result
        #     print(author_name[i] + "    " + str(dev_result) + " / " + str(len(dev_set)) + " correct")





if __name__ == "__main__":
    print("training... (this may take a while)")
    main()
