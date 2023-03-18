import os.path
import argparse
from nltk import data
import nltk
import numpy as np

# CKY Parser

class Node:
    """
    Used for storing information about a non-terminal symbol. A node can have a maximum of two
    children because of the CNF of the grammar.
    It is possible though that there are multiple parses of a sentence. In this case information
    about an alternative child is stored in self.child1 or self.child2 (the parser will decide
    where according to the ambiguous rule).
    Either child1 is a terminal symbol passed as string, or both children are Nodes.
    """

    def __init__(self, symbol, child1, child2=None):
        self.symbol = symbol
        self.child1 = child1
        self.child2 = child2

    def __repr__(self):
        """
        :return: the string representation of a Node object.
        """
        return self.symbol

def read_grammar(grammar_file):
    cfgrammar = []
    with open(grammar_file) as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            cfgrammar.append(line.replace("->", "").split())
    return cfgrammar

def cnf_convert(grammar):
    cnf_grammar = []
    used_symbols = []

    for rule in grammar:
        new_rules = []
        if len(rule) > 2: # not unit production
            # Case: A -> X a.
            terminals = []
            for i, r in enumerate(rule):
                if r[0].islower(): # terminals starts with lowercase letter
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

def add_unitprod(cell, cnf_grammar):
    for c in cell:
        for g in cnf_grammar:
            if len(g) == 2 and c[0]==g[1] and g not in cell:
                cell.append(g)

def cyk_parse(sentence, cnf_grammar):
    print(sentence)
    # print('')
    words = sentence.split()
    m = len(words)
    table = np.empty((m,m), dtype = object)
    parse_table = [[[] for x in range(m - y)] for y in range(m)]
    for j in range(0, m): # add diagnoal of the table
        table[j][j] = [g for g in cnf_grammar if words[j] in g]
        add_unitprod(table[j][j], cnf_grammar)
        for i in range(j-1, -1, -1):
            table[i][j] = []
            for k in range(i, j):
                b = [r1[0] for r1 in table[i][k]]
                c = [r2[0] for r2 in table[k+1][j]]

                for g1 in cnf_grammar:
                    if len(g1) == 3 and ((g1[1] in b) and (g1[2] in c)):
                        table[i][j].append(g1)
                    add_unitprod(table[i][j], cnf_grammar)
    return table


def recognize_tree(table):
    m,m = table.shape
    flg = 0
    for s in table[0,m]:
        if 'S' in s:
            flg = 1

    return flg


def read_to_tree(table):
    #
    pass

def print_tree():
    return 0

def main():
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("grammar")
    # # argparser.add_argument("sentence")
    # args = argparser.parse_args()
    # grammar_file = args.grammar
    # sentence = args.sentence

    grammar_file = "l1.cfg"
    setence = "i book a flight"
    cnf_grammar = cnf_convert(read_grammar(grammar_file))
    tree = cyk_parse(setence,cnf_grammar)
    print(tree[0,3])



# gm = data.load(grammar_file)
if __name__ == '__main__':
    main()
