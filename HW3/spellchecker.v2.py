import time
import numpy as np
import sys
import re

#### for numbers, don't change
#### for abbreviations, check by a abbreviation dictionary 

startTime = time.time()

def tokenization(line):
    """
    adapted from Figure 3.22 in textbook
    """
    # define regex patterns
    orig_line = line
    letters_regex = r"[A-za-z0-9]"
    noletter = r"^[A-za-z0-9]"
    alwayssep = r"[?!()“;/|']"
    clitics = r"'|:|-|'S|'D|'M|'LL|'RE|'VE|N'T|'s|'d|'m|'ll|'re|'ve|n't"
    abbr = ["Co.", "Corp.", "vs.", "e.g.", "etc.", "ex.", "cf.",
            "eg.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.",
            "Sept.", "Oct.", "Nov.", "Dec.", "jan.", "feb.", "mar.",
            "apr.", "jun.", "jul.", "aug.", "sept.", "oct.", "nov.",
            "dec.", "ed.", "eds.", "repr.", "trans.", "vol.", "vols.",
            "rev.", "est.", "b.", "m.", "bur.", "d.", "r.", "M.", "Dept.",
            "MM.", "U.", "Mr.", "Jr.", "Ms.", "Mme.", "Mrs.", "Dr.",
            "Ph.D."]
    # put white space around unambiguous separators
    line = re.sub("("+alwayssep+")",r" \1 ", line)
    # put whitespace around commas that aren't inside numbers
    line = re.sub(r"(^[0-9]),",r"\1 , ", line)
    line = re.sub(r",(^[0-9])", r" , \1", line)

    # distinguish singlequotes from apostrophes by segmenting off single quotes not preceded by letter 
    line = re.sub(r"^(')", r"\1 ", line)
    line = re.sub("("+ noletter + ")'", r"\1 '", line)

    #segment off unambiguous word-final clitics and punctuation
    line = re.sub("("+ clitics + ")$", r" \1", line)
    line = re.sub("(" + clitics + ")(" + noletter + ")", r" \1 \2", line)

    #deal with periods. For each possible word
    #this is a list
    possible_words = line.split()
    #initialize a token list
    tokens = []
    for word in possible_words:
        #if it ends in a period, and isn't in abbr and isn't a sequence of periods (U.S.) and does not resample an abbreviation (Inc.)
        if re.search(letters_regex + r"\.", word) and not (re.search(r"^([A-Za-z]\.([A-Za-z]\.)+|[A-Z][bcdfghj-nptvxz]+\.)$", word)) and (word not in abbr):
            #then segment off the period  
            word = re.sub(r"\.+$",r"", word)
        tokens.append(word)

    return tokens


def edit_distance(source, target):
    """
    Calculate distance after tokenization
    algorithm using dynamic programming borrowed from textbook Figure 3.25
    """
    #define cost
    del_cost, ins_cost = 1, 1
    sub_cost = lambda a,b: 0 if (a == b) else 2
    n,m = len(source),len(target) 
    #create a matrix storing distance of two strings
    #initialize 0 at the two empty strings
    distance  = np.zeros([n+1, m+1])
    distance[0][0] = 0
    #iterating rows from bottom left
    for i in range(n-1, -1, -1):
        distance[i][0] = distance[i+1][0] + del_cost
    #iterating columns from bottom left
    for j in range(1, m+1):
        distance[n][j] = distance[n][j-1] + ins_cost
    #print(distance)

    #iterating inner space of the distance matrix
    for i in range(n-1, -1, -1):
        for j in range(1, m+1):
            distance[i][j] = min(distance[i+1][j] + del_cost,
                                 distance[i][j-1] + ins_cost,
                                 distance[i+1][j-1] + sub_cost(source[n-1-i], target[j-1]))
    #print(distance)
    return distance[0][m]



def suggest_best(token):
    suggestion = token
    # Filtering special case
    # Token contains not word (U.S)
    if re.match(r'[A-Za-z]+', token):
        dic_dir = './words'
        with open(dic_dir) as f:
            dic = f.read().split('\n')
        ### filtering the dictionary using the length 
        l = len(token)
        lb = max(0, l-2)
        ub = l+2 
        subdic = [x for x in dic if len(x) >= lb and len(x) <= ub] 
    
        ### filtering dictionary using first letter with possible misspelling based on keyboard location
        keyboard = {'q':['w','a'], 'w':['q','e','s'], 'e':['w','d','r'],'r':['e','f','t'],'t':['r','g','y'],
                'y':['t','h','u'], 'u':['y','j','i'], 'i':['u','k','o'], 'o':['i','l','p'], 'p':['o'],
                'a':['q','s','z'], 's':['a','w','d','x'], 'd':['e','s','f','c'],'f':['r','d','g','v'],
                'g':['t','f','h','b'],'h':['y','g','j','n'],'j':['u','h','k','m'],'k':['i','j','l'],'l':['k','o'],
                'z':['a','x'],'x':['z','s','c'], 'c':['x','d','v'], 'v':['c','f','b'],'b':['g','v','n'],
                'n':['b','h','m'],'m':['n','j']}
        letters_lowercase = (token[0].lower(),) + tuple(keyboard[token[0].lower()]) 
        letters_uppercase = (token[0].upper(),) + tuple(list(map(lambda x: x.upper(), keyboard[token[0].lower()])))
        letters = letters_lowercase + letters_uppercase
        subdic = [d for d in subdic if d.startswith(letters)]
        #print(len(subdic))
        if (token in subdic) or (token.lower() in subdic):
            return None
        
        similarity = dict(zip(subdic,[edit_distance(x, token) for x in subdic]))
        threshold = 5
        similarity = {key:value for key, value in similarity.items() if value <= threshold}
        if len(similarity) > 0:
            suggestions = [k[0] for k in sorted(similarity.items(), key=lambda x: x[1])]
            if len(suggestions) > 3:
                suggestions = suggestions[:3]
            suggestion = suggestions[:1]
        else:
            suggestions = ["No suggestions from this corpus!"]
            suggestion = token
        
    return suggestion

    






def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace("' ", "'")
    return step6.strip()







def main():
    f = open(sys.argv[1])
    for line in f:
        tokens = tokenization(line)
        new_line = []
        for token in tokens:
            print(suggest_best(token))
            # new_word = suggest_best(token)
            # new_line.append(new_word)
            new_line.append(token)
        # print(untokenize(new_line))

        
    #checked with book's example
    #print(edit_distance('intention','execution'))  
    # print(suggest("helloa"))


if __name__ == "__main__":
    main()
        
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
