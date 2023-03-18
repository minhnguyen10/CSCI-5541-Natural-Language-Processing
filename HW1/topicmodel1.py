import sys
import re

# from nltk.corpus import stopwords

# My approach is to find the top 5 most occured words in the text 
# after cleaning it using regex and modified version of stopword list in NLTK

def best_words(f, topwords):
    data = []
    for line in f:
        words = re.findall( r'\w+|[^\s\w]+', line)
        for word in words:
            data.append(word)
    clean_data = clean(data)
    word_dict = word_count(clean_data)
    best_words = sorted(word_dict, key=word_dict.get, reverse=True)[:topwords]
    return best_words

def clean(data):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
                   'ourselves', 'you', "you're", "you've", "you'll", 
                   "you'd", 'your', 'yours', 'yourself', 'yourselves', 
                   'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                   'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 
                   'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                   'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
                   'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                   'have', 'has', 'had', 'having', 'do', 'does', 'did', 
                   'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                   'because', 'as', 'until', 'while', 'of', 'at', 'by', 
                   'for', 'with', 'about', 'against', 'between', 'into', 
                   'through', 'during', 'before', 'after', 'above', 'below', 
                   'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
                   'over', 'under', 'again', 'further', 'then', 'once', 
                   'here', 'there', 'when', 'where', 'why', 'how', 'all', 
                   'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                   'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
                   'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
                   'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 
                   'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
                   'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
                   "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
                   "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
                   "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', 
                   "weren't", 'won', "won't", 'wouldn', "wouldn't", "may", "must", "shall"]
    # stop_words = set(stopwords.words('english'))
    r = re.compile("[\w\s]")
    clean_data = list(filter(r.match, data))
    filtered_data = []
    for word in clean_data:
        word = word.lower()
        if word not in stop_words:
            filtered_data.append(word)
    
    return filtered_data

def word_count(data):
    word_dict = {}
    for word in data:
        if word in word_dict: 
            word_dict[word] += 1
        else: 
            word_dict[word] = 1
    print(word_dict)
    return word_dict

def main():
    topwords = 5 # top 5 word
    # f = open('text.txt',"r")
    f = open(sys.argv[1])
    for word in best_words(f, topwords):
        print(word)
    f.close()

if __name__ == '__main__':
   main()