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
    #words = tokens
    tok = [word.lower() for word in words]
    return tok


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

# Models
def get_ngram(train_set, n = 2):
    """
    calculate n-gram model and return the training vocabulary
    """
    vocab = set(words.words())
    ngramset = []
    n1gramset = []
    word_list = []
    for sentence in train_set:
        starts = list(np.repeat('<START>',n-1))
        toks = word_tokenization(sentence)
        toks = [class_UNK(tok) if tok not in vocab else tok for tok in toks ]  ### we can replace <UNK> with some UNK classification function
        toks = starts + toks

        ngram = [tuple(toks[i:i+n]) if n > 1 else toks[i] for i in range(len(toks) - n+1)]
        n1gram = [tuple(toks[i:i+n-1]) if n > 2 else toks[i] for i in range(min(len(toks) - n+2,len(toks)))]
        ngramset.extend(ngram)
        n1gramset.extend(n1gram)
        word_list.extend([toks[i] for i in range(len(toks))])

    ngramdict = Counter(ngramset)
    n1gramdict = Counter(n1gramset)

    ff = Counter(ngramdict.values())
    ff1 = Counter(n1gramdict.values())
    train_vocab = set(word_list)
    N = len(train_vocab)
    if n > 1:
        n0 = N ** n  - sum(ff.values())
        ff[0] = n0
        n0 = N ** (n-1)  - sum(ff1.values())
        ff1[0] = n0
      ## dictionary stores frequencies of all ngram
    return [ngramdict, n1gramdict], [ff, ff1], train_vocab






# Smoothing methods
# try GT-discount first
def smooth(c, model,fftable,k=6):
    #n = len(list(model.keys())[0])
    #print(fftable.get(0))
    ### don't smooth for unigram model
    if c < k:
        #print(c)
        #print(fftable[0])
        c = (c+1) * fftable.get(c+1)/fftable.get(c)
    return c

def calculate_prob(models, train_vocab, dev_set, ff):
    vocab = set(words.words())
    #all_keys = list(models[0][0].keys())
    ### get ngram parametr n
    n = len(next(iter(models[0])))
    #N = len(train_vocab)
    #print(n)
    probs = []
    for sentence in dev_set:
        starts = list(np.repeat('<START>',n-1))
        toks = word_tokenization(sentence)
        toks = [class_UNK(tok) if (tok not in vocab) or (tok not in train_vocab)  else tok for tok in toks]
        toks = starts + toks
        prob = 0
        #print(toks)
        ### calculate prob
        for i in range(len(toks) - n + 1):
            if n==1:
                inquery = toks[i]
                nominator =  models[0].get(inquery, 0)
                denominator = 1
                p = nominator/denominator

            else:
                ngram_inquery = tuple(toks[i:i+n])
                nominator = models[0].get(ngram_inquery, 0)
                if n == 2:
                    n1gram_inquery = ngram_inquery[0]
                    #print(ngram_inquery, n1gram_inquery)
                    denominator = models[1].get(n1gram_inquery, 0)
                    nominator = smooth(nominator, model=models[0], fftable = ff[0])
                    p = nominator/denominator
                else:
                    n1gram_inquery = ngram_inquery[:-1]
                    #print(ngram_inquery, n1gram_inquery)
                    nominator = models[0].get(ngram_inquery, 0)
                    denominator = models[1].get(n1gram_inquery, 0)
                    nominator = smooth(nominator, model=models[0], fftable = ff[0])
                    denominator = smooth(denominator, model= models[1], fftable = ff[1])
                    p = nominator/denominator

            ### log transformation to overcome underflow
            #print(nominator, denominator)
            prob = prob  + np.log(p)
            ### probability for model for each sentence
        probs.append(prob)
    return probs




# Predict
def predict(n_models, train_vocab, dev_set, ffs):
    results = []
    for i in range(len(n_models)):
        probs = calculate_prob(n_models[i], train_vocab[i], dev_set, ffs[i])
        results.append(probs)
    results = np.vstack(results)
    #print(results.shape)
    return results


# Main class that read option
def main():
    ## initialize parser
    parser = argparse.ArgumentParser()
    ## add argument
    parser.add_argument("input")
    parser.add_argument( "-test", help = "output filename")
    parser.add_argument("-n", help = "use n-gram")
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

        #for line in text:
                # tokens = sent_tokenize(line)
            # print(summarization.textcleaner.clean_text_by_word(line))
            #print("")
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

        n_models = []
        ffs = []
        train_vocabs = []
        for i in range(len(train_sets)):
            print("Load " + author_name[i] +" ...")
            if args.n:
                models, ff, train_vocab = get_ngram(train_sets[i], n = int(args.n))
            else:
                models, ff, train_vocab = get_ngram(train_sets[i])
            n_models.append(models)
            ffs.append(ff)
            train_vocabs.append(train_vocab)

        print('DONE TRAINING!')
        print("Results on a specific dev set:")
        dev_probs = predict(n_models, train_vocabs,dev_sentences_clean, ffs)
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
            # print(len(train_set))
            # print(dev_set_clean)


        # Building different n-gram model
        n_models = []
        ffs = []
        train_vocabs = []
        for i in range(len(train_sets)):
            # train the model, return the model
            if args.n:
                models, ff, train_vocab = get_ngram(train_sets[i], n = int(args.n))
            else:
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
            dev_probs = predict(n_models, train_vocabs,dev_sets[i], ffs)
            idx = np.argmax(dev_probs, axis = 0)
            pred_author = [k==i for k in idx]
            dev_result = sum(pred_author)

            # print result
            print(author_name[i] + "    " + str(dev_result) + " / " + str(len(dev_sets[i])) +
            " = " + str(round((dev_result/len(dev_sets[i]))*100, 2)) + "%"
            + " correct")





if __name__ == "__main__":
    print("training... (this may take a while)")
    main()



executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
