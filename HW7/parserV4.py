import argparse
from nltk.tree import *
import numpy as np

# CKY Parser
class Node:
    def __init__(self, nt, left, right = None):
        self.nt = nt
        self.left = left
        self.right = right

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
                    print((r, i))
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
            if len(g) == 2 and c.label()==g[1]:
                gnode = Tree(g[0],[c])
                cell.append(gnode)

def cky_parse(sentence, cnf_grammar):
    words = sentence.split()
    m = len(words)
    table = np.empty((m,m), dtype = object)
    for j in range(0, m): # add diagnoal of the table
        table[j][j] = [Tree(g[0],[words[j]]) for g in cnf_grammar if words[j] in g]
        add_unitprod(table[j][j], cnf_grammar)
        for i in range(j-1, -1, -1):
            table[i][j] = []
            for k in range(i, j):
                # get all labels from previous cell
                b = [r1.label() for r1 in table[i][k]]
                c = [r2.label() for r2 in table[k+1][j]]

                for g1 in cnf_grammar:
                    if len(g1) == 3 and ((g1[1] in b) and (g1[2] in c)):

                        # find corresponding rules
                        lhs_list = []
                        for r1 in table[i][k]:
                            if r1.label() == g1[1]:
                                lhs_list.append(r1)

                        
                        rhs_list = []
                        for r2 in table[k+1][j]:
                            if r2.label() == g1[2]:
                                rhs_list.append(r2)
                        
                        # if len(rhs_list) > 0:
                        for lhs in lhs_list:
                            for rhs in rhs_list:
                                table[i][j].append(Tree(g1[0],[lhs,rhs]))
            add_unitprod(table[i][j], cnf_grammar)


    return table

def recognize_tree(table):
    m,m = table.shape
    flg = 0
    for s in table[0,m-1]:
        if 'S' == s.label():
            flg = 1
            return flg
    return flg



def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("grammar")
    argparser.add_argument("sentence")
    args = argparser.parse_args()
    grammar_file = args.grammar
    sentence = args.sentence


    cnf_grammar = cnf_convert(read_grammar(grammar_file))
    table = cky_parse(sentence, cnf_grammar)
    
    for i in range(0,6):
        for j in range(0,6):
            print('---',str(i),'--',str(j),'----------------------')
            print(table[i,j])
            # print('----------------------')

    if recognize_tree(table):
        m,m = table.shape
        for s in table[0,m-1]:
            if s.label() == 'S':
                #print(s)
                s.pretty_print()
                print("-----")
    else:
        print("No parse tree for this sentence in the grammar")

    # print(table)

# gm = data.load(grammar_file)
if __name__ == '__main__':
    main()
