import nltk
import string
from collections import Counter
import pickle
from nltk.corpus import words
# nltk.download('brown')
from nltk import WordNetLemmatizer




def model_build(train_corp):
    ### add start token to each sentence
    train_corp = [[('<START>','<START>')] + sent + [('<STOP>','<STOP>')] for sent in train_corp]
    all_tokens = [entry[0]  for sent in train_corp for entry in sent]
    all_tokens = determ_UNK(all_tokens)
    all_tags = [entry[1] for sent in train_corp for entry in sent]
    all_bigram_tags = [tuple([all_tags[i],all_tags[i+1]]) for i in range(len(all_tags)-1)]
    ugram_tag = Counter(all_tags)
    bgram_tag = Counter(all_bigram_tags)
    ugram_tw = Counter(zip(all_tags, all_tokens))

    return ugram_tag, bgram_tag, ugram_tw

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


# get unknown word
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


def main():

    print("loading training set...")

    train_corp = nltk.corpus.brown.tagged_sents() #[1:10]

    ugram_tag, bgram_tag, ugram_tw = model_build(train_corp)

    # print(ugram_tag)
    # print(bgram_tag)
    # print(ugram_wt)

    transition_mat, emission_mat = calc_prob(ugram_tag, bgram_tag, ugram_tw)
    transition_mat.pop('<STOP>', None)
    model = [transition_mat, emission_mat]


    print("writting to model.dat...")
    with open("model.dat", "wb") as out:
         pickle.dump(model, out)


if __name__ == '__main__':
    main()
