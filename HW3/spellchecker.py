import argparse
import os 
import time
import numpy as np
import sys
import re


##########################################################
# authors: Zhecheng Sheng, Minh Nguyen                    
# usage: spellchecker.py [-h] [-o OUTPUT] [-x] input     #
#                                                        #
# positional arguments:                                  #
#  input                                                 #
#                                                        #
# optional arguments:                                    #   
#  -h, --help            show this help message and exit #
#  -o OUTPUT, --output OUTPUT                            #
#                        output filename                 #
#  -x, --interactive     interactive model               #
#                                                        #
##########################################################



#### for numbers, don't change
#### for abbreviations, check by a abbreviation dictionary 

startTime = time.time()
dic_dir = './words'

def tokenization(line):
    """
    adapted from Figure 3.22 in textbook
    """
    # define regex patterns
    letters_regex = r"[A-za-z0-9]"
    noletter = r"^[A-za-z0-9]"
    alwayssep = r"[?!()“;/|']"
    clitics = r"'|:|-|'S|'D|'M|'LL|'RE|'VE|N'T|'s|'d|'m|'ll|'re|'ve|n't"
    abbr = ["Co.", "Corp.", "vs.", "e.g.", "etc.", "ex.", "cf.",
            "eg.", "ed.", "eds.", "repr.", "trans.", "vol.", "vols.",
            "rev.", "est.", "b.", "m.", "bur.", "d.", "r.", "M.", "Dept.",
            "MM.", "U.", "Mr.", "Jr.", "Ms.", "Mme.", "Mrs.", "Dr.",
            "Ph.D."]
    states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']


    # put white space around unambiguous separators
    line = re.sub("("+alwayssep+")",r" \1 ", line)
    # put whitespace around commas that aren't inside numbers
    line = re.sub(r"([^0-9]),",r"\1 , ", line)
    line = re.sub(r",([^0-9])", r" , \1", line)

    # distinguish singlequotes from apostrophes by segmenting off single quotes not preceded by letter 
    line = re.sub(r"^(')", r"\1 ", line)
    line = re.sub("("+ noletter + ")'", r"\1 '", line)

    #segment off unambiguous word-final clitics and punctuation
    line = re.sub("("+ clitics + ")$", r" \1", line)
    line = re.sub("(" + clitics + ")(" + noletter + ")", r" \1 \2", line)

    #deal with name (e.g John Doe)
    # line = re.sub(r' [A-Za-z]+((\s)?((\'|\-|\.)?([A-Za-z])+))', 'name\1', line)
    # line = re.sub(r' [A-Za-z]+((\s)?((\'|\-|\.)?([A-Za-z])+))', 'name\1', line)
    #deal with periods. For each possible word
    #this is a list
    possible_words = line.split()
    #initialize a token list
    tokens = []
    for word in possible_words:
        #if it ends in a period, and isn't in abbr and isn't a sequence of periods (U.S.) and does not resample an abbreviation (Inc.)
        if re.search(letters_regex + r"\.", word) and not (re.search(r"^([A-Za-z]\.([A-Za-z]\.)+|[A-Z][bcdfghj-nptvxz]+\.)$", word)) and (word not in abbr) and (word not in states):
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

def stem(token):
    """functions that stems token"""
    pos_token = []
    if token.endswith("s"):
        stoken = re.sub(r"es$","", token)
        pos_token.append(stoken)
        stoken = re.sub(r"s$","", token)
        pos_token.append(stoken)
        stoken = re.sub(r"ies$","y", token)
        pos_token.append(stoken)

    elif token.endswith("d"):
        stoken = re.sub(r"ed$","", token)
        pos_token.append(stoken)
        stoken = re.sub(r"d$","", token)
        pos_token.append(stoken)
        stoken = re.sub(r"ied$","y", token)
        pos_token.append(stoken)
            
    else:
        pass

    return pos_token




def suggest(token):
    #dic_dir = './words' #
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
    ## also consider singular, plural and past tense, most general cases
    if (token in subdic) or (token.lower() in subdic) or (any(x in stem(token) for x in subdic)) or (any(x in list(map(lambda s: s.lower(),stem(token))) for x in subdic)):
        return None
    
    similarity = dict(zip(subdic,[edit_distance(x, token) for x in subdic]))
    threshold = 5 
    similarity = {key:value for key, value in similarity.items() if value <= threshold}

    if len(similarity) > 0:
        suggestions = [k[0] for k in sorted(similarity.items(), key=lambda x: x[1])]
        if len(suggestions) > 3:
            suggestions = suggestions[:3]
    else:
        suggestions = ["No suggestions from this corpus!"]

    return suggestions

    

def replace(corrected, orig, line):
    """
    Function that corrects wrong word
    """
    ### keep case the same
    if orig.isupper():
        corrected = corrected.upper()
    elif ord(orig[0]) < 97:
        corrected = corrected[0].upper() + corrected[1:]

    return line.replace(orig, corrected)




def main():
    ## initialize parser
    parser = argparse.ArgumentParser()
    ## add argument
    parser.add_argument("input")
    parser.add_argument( "-o", "--output", help = "output filename")
    parser.add_argument( "-x", "--interactive", help = "interactive model", action="store_true")
    args = parser.parse_args()



    f = open(args.input)
    output = []
    linenumber = 0
    for line in f:
        tokens = tokenization(line)
        for token in tokens:
            ### only care letter words and assume all the abbreviations are correct
            if re.fullmatch(r"[A-Za-z]+", token):
                orig_word = token
                suggestions = suggest(token)
                if suggestions is None:
                    continue
                else:
                    print("The mis-spelled word %s appears in line %d" %(token,linenumber+1))
                    print("the possible suggestions from the dictionary are:")
                    print(*suggestions, sep = ',')
                    correct = suggestions[0]
                    if args.interactive:
                        choice = input("Choose one to replace or type 0 to skip:\n")
                        choice = int(choice)
                        if choice > 0:
                            correct = suggestions[choice-1]
                            line = replace(correct, token, line)
                        else:
                            pass
                    elif correct != "No suggestions from this dictionary!":
                        line = replace(correct, token, line)

                    print("------")
            else:
                continue
        # Handle multiple blanks
        line = re.sub(r' +', r' ', line);
        output.append(line)
        linenumber = linenumber + 1 

    ### close connection
    f.close()
    if args.output:
        with open(args.output, 'w') as ofile:
            ofile.writelines(output)


    else:
        with open("corrected_"+ os.path.basename(args.input), 'w') as ofile:
            ofile.writelines(output)

if __name__ == "__main__":
    main()
        
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
