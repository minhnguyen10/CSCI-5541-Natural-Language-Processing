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

startTime = time.time()


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
def get_ngram(train_set, n = 2):
    """
    calculate n-gram model and return the training vocabulary
    """
    vocab = set(words.words())
    ngramset = []
    n1gramset = []
    for sentence in train_set:
        starts = list(np.repeat('<START>',n-1))
        toks = word_tokenization(sentence)
        toks = ['<UNK>' if tok not in vocab else tok for tok in toks ]  ### we can replace <UNK> with some UNK classification function
        toks = starts + toks
        ngram = [(toks[i],toks[i+n-1]) if n > 1 else toks[i] for i in range(len(toks) - n+1)]
        n1gram = [(toks[i],toks[i+n-2]) if n > 2 else toks[i] for i in range(len(toks) - n+2)]
        ngramset.extend(ngram)
        n1gramset.extend(n1gram)

    ngramdict = Counter(ngramset)
    n1gramdict = Counter(n1gramset)

    ff = Counter(ngramdict)
    ff1 = Counter(n1gramdict)
    train_vocab = set([ x for a in list(ngramdict.keys()) for x in a])
      ## dictionary stores frequencies of all ngram
    return [ngramdict, n1gramdict], [ff, ff1], train_vocab





# Smoothing methods
# try GT-discount first
def smooth(c, model,fftable, train_vocab, k=6):
    n = len(list(model.keys())[0])
    N = len(train_vocab)
    fftable[0] = N**n  - sum(fftable.values())
    if c < k:
        c = (c+1) * fftable.get(c+1, 0.5)/fftable.get(c, 0.5)
    return c

def calculate_prob(models, train_vocab, dev_set, ff):
    vocab = set(words.words())
    #all_keys = list(models[0][0].keys())
    n = 2
    probs = []
    for sentence in dev_set:
        starts = list(np.repeat('<START>',n-1))
        toks = word_tokenization(sentence)
        toks = starts + toks
        toks = ['<UNK>' if (tok not in vocab) and (tok not in train_vocab)  else tok for tok in toks]
        prob = 0

        ### calculate prob
        for i in range(len(toks) - n + 1):
            if n==1:
                inquery = toks[i]
                nominator =  models[0].get(inquery, 0)
                denominator = 1
                ### need smooth for nominator
                nominator = smooth(nominator, k = 6, model= models)
            else:
                ngram_inquery = tuple(toks[i:i+n])
                if n == 2:
                    n1gram_inquery = ngram_inquery[0]
                else:
                    n1gram_inquery = ngram_inquery[:-1]
                #print(ngram_inquery, n1gram_inquery)
                nominator = models[0].get(ngram_inquery, 1)
                denominator = models[1].get(n1gram_inquery, 1)
                #print(nominator, denominator)
                ### need smooth for nominator and denominator
                nominator = smooth(nominator, model=models[0], train_vocab = train_vocab, fftable = ff[0])
                denominator = smooth(denominator, model= models[1], train_vocab = train_vocab, fftable = ff[1])
            ### log transformation to overcome underflow
            #print(nominator, denominator)
            prob = prob  + np.log(nominator/denominator)
            ### probability for model for each sentence
        probs.append(prob)
    return probs




# Predict
def predict(n_models, train_vocab, dev_set, ffs):
    results = []
    for i in range(len(n_models)):
        probs = calculate_prob(n_models[i], dev_set, train_vocab, ffs[i])
        results.append(probs)
    results = np.vstack(results)
    print(results.shape)
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
        text = f.read()

        #for line in text:
                # tokens = sent_tokenize(line)
            # print(summarization.textcleaner.clean_text_by_word(line))
            #print("")


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
            #for sentence in dev_set_clean:
                #print("")
                #print(word_tokenization(sentence))

            # print(len(sentences))
        # print(len(train_sets))
        # print(dev_sets)


        # Building different n-gram model
        n_models = []
        ffs = []
        train_vocabs = []
        for i in range(len(train_sets)):
            # train the model, return the model
            models, ff, train_vocab = get_ngram(train_sets[i])
            n_models.append(models)
            ffs.append(ff)
            train_vocabs.append(train_vocab)
        print('DONE TRAINING!')
        # Getting result from each model, print the highest probabilty as the lable
        print("Results on dev set:")
        for i in range(len(dev_sets)):
            # predict the development set using all training models
            # (we need to use all ngram models to find the one with highest prob)
            dev_probs = predict(n_models, dev_sets[i], train_vocabs[i], ffs)
            idx = np.argmax(dev_probs, axis = 0)
            print(idx)
            pred_author = [k==i for k in idx]
            dev_result = sum(pred_author)

            # print result
            print(author_name[i] + "    " + str(dev_result) + " / " + str(len(dev_sets[i])) + " correct")





if __name__ == "__main__":
    print("training... (this may take a while)")
    main()



executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
