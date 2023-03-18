import numpy as np
from collections import defaultdict
import pickle
import argparse
from nltk.corpus import words, brown
from nltk import WordNetLemmatizer
import string



def determ_UNK(tokens):
    vocab = set(words.words())
    ans = []
    wn = WordNetLemmatizer()
    for token in tokens:
        token = wn.lemmatize(token)
        if (token.lower() not in vocab) and (token not in vocab)  and (token not in ['<START>','<STOP>']) and (token not in string.punctuation):
            ans.append(class_UNK(token))
        else:
            ans.append(token)
    return ans

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

def get_viterbi( tran_model, emi_model, W, Q):

    ## For simplicity, assume words and punctuations are space delimited
    W = "<START> " + W + " <STOP>"
    tokens = W.split()
    tokens = determ_UNK(tokens)
    q0 = '<START>'
    qf = '<STOP>'
    Vit = defaultdict(dict)
    Back = defaultdict(dict)

    # Initiation Step
    for s in Q:
        if (s in tran_model[q0]) and (tokens[1] in emi_model[s]):
            Vit[1][s] = np.log(tran_model[q0][s]) + np.log(emi_model[s][tokens[1]])

        else:
            Vit[1][s] = np.log(1e-15)
        Back[1][s] = q0

    # Recursion Step
    for t in range(2,len(tokens)-1):
        for s in Q:
            Vit[t][s] = - np.inf
            for sprev in Q:
                if (s in tran_model[sprev]) and (tokens[t] in emi_model[s]):
                    temp = Vit[t-1][sprev] + np.log(tran_model[sprev][s]) +  np.log(emi_model[s][tokens[t]])
                    # print("here")
                else:
                    temp = Vit[t-1][sprev] + np.log(1e-15)
                    #print(temp)
                if temp > Vit[t][s]:
                    Vit[t][s] = temp
                    Back[t][s] = sprev

    # Termination Step
    lastV = - np.inf
    for s in Q:
        if qf in tran_model[s]:
            ### only transition no emission
            temp = Vit[len(tokens)-2][s] +  np.log(tran_model[s][qf])
        else:
            temp = Vit[len(tokens)-2][s] + np.log(1e-15)

        if temp > lastV:
            lastV = temp
            lastq = s


    # Trace back for the best tag sequences using Back dictionary
    sequences = []
    sequences.append(lastq)
    for i in range(len(tokens)-2):
        sprev  = Back[len(tokens)-2-i][sequences[i]]
        sequences.append(sprev)
    # print(sequences)
    return sequences[::-1][1:]

def main():
        ## initialize parser
    parser = argparse.ArgumentParser()
    ## add argument
    parser.add_argument("model")
    parser.add_argument("example")
    args = parser.parse_args()

    # read model
    #print("loading models...")
    with open(args.model, "rb") as fb:
        tran_model, emi_model = pickle.load(fb)

    # read example
    #print("reading text...")
    with open(args.example) as f:
        seq = f.read()

    print("getting tagset...")
    #Q = set([t for sent in brown.tagged_sents() for w,t in sent])
    Q = set()
    sents = brown.tagged_sents()
    for sent in sents:
        for w, t in sent:
            Q.add(t)

    print("calculating tag sequence...")
    tag_seq = get_viterbi(tran_model, emi_model, seq, Q)
    print("The BEST tag sequence for: \n%s%s" %(seq, tag_seq))

if __name__ == '__main__':
    main()
