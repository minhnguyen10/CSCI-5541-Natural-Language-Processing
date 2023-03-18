import os.path
import argparse
from nltk import data

RULE_DICT = {}

def read_grammar(grammar_file):
    cfgrammar = []
    with open(grammar_file) as file:
        lines = file.readlines()
        for line in lines:
            cfgrammar.append(line.replace("->", "").split())
    return cfgrammar

def cnf_convert(grammar):
    cnf_grammar = []
    used_symbols = []

    for rule in grammar:
        new_rules = []
        if len(rule) > 2:    
            # Case: A -> X a.
            terminals = []
            for i, r in enumerate(rule):
                if r[0] == "'":
                    terminals.append((r, i))       
            if len(terminals) > 0:
                for item in terminals:
                    idx = 0
                    while f"{rule[0]}{str(idx)}" in used_symbols:
                        idx += 1
                    new_rules.append([f"{rule[0]}{str(idx)}", item[0]])
                    rule[item[1]] = f"{rule[0]}{str(idx)}"      
                    used_symbols.append(f"{rule[0]}{str(idx)}")              
            # Case A -> X B C [...]
            while len(rule) > 3:
                idx = 0
                while f"{rule[0]}{str(idx)}" in used_symbols:
                    idx += 1   
                used_symbols.append(f"{rule[0]}{str(idx)}")  
                new_rules.append([f"{rule[0]}{str(idx)}", rule[1], rule[2]])
                rule = [rule[0]] + [f"{rule[0]}{str(idx)}"] + rule[3:]
        cnf_grammar.append(rule)
        if len(new_rules) > 0:
            cnf_grammar.extend(new_rules)
    return cnf_grammar


def cyk_parse(cnf_grammar):
    return 0

def read_to_tree(table):
    return 0

def print_tree():
    return 0

def main():
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("grammar")
    # # argparser.add_argument("sentence")
    # args = argparser.parse_args()                                   
    # grammar_file = args.grammar
    # sentence = args.sentence
    
    grammar_file = "grammar2.cfg"
    setence = "i called mom at home"
    cnf_grammar = cnf_convert(read_grammar(grammar_file))
    # tree = cyk_parse(cnf_grammar)
    print(cnf_grammar)


# gm = data.load(grammar_file)
if __name__ == '__main__':
    main()