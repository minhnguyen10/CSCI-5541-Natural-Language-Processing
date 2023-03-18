import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from random import randint
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer


def get_unigrams(corpus):
	unigrams={}
	for sentence in corpus.split('\n'):
		tokens=word_tokenize(sentence)
		for k in tokens:
			if k in unigrams:
				unigrams[k]+=1
			else:
				unigrams[k]=1
	return unigrams

def get_bigrams(corpus):
	bigrams={}
	for sentence in corpus.split('\n'):
		tokens=word_tokenize(sentence)
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

def main():
    f = open('exam.txt')
    text = f.read()
    # for line in :
    unigram = get_unigrams(text)
    bigrams = get_bigrams(text)
    print(len(unigram))



if __name__ == '__main__':
  main()