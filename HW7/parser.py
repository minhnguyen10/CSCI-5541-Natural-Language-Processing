import os.path
import argparse
from nltk import data

RULE_DICT = {}

def read_grammar(grammar_file):
    grammar_list = []
    with open(grammar_file) as cfg:
        lines = cfg.readlines()
        for line in lines:
            grammar_list.append(line.replace("->", "").split())
    return grammar_list

def add_rule(rule):
    global RULE_DICT

    if rule[0] not in RULE_DICT:
        RULE_DICT[rule[0]] = []
    RULE_DICT[rule[0]].append(rule[1:])

def convert_grammar(grammar):
    # Remove all the productions of the type A -> X B C or A -> B a.
    global RULE_DICT
    # current_rule = {}
    cnf_grammar = []
    unit_productions = []
    # unit_productions, result = [], []
    res_append = cnf_grammar.append
    index = 0

    for rule in grammar:
        new_rules = []
        # if len(rule) == 2 and rule[1][0] != "'":
        #     # Rule is in form A -> X, so back it up for later and continue with the next rule.
        #     unit_productions.append(rule)
        #     add_rule(rule)
        #     continue
        # elif len(rule) > 2:
        if len(rule) > 2:    
             # or A -> X a.
            terminals = [(item, i) for i, item in enumerate(rule) if item[0] == "'"]
            if terminals:
                for item in terminals:
                    # Create a new non terminal symbol and replace the terminal symbol with it.
                    # The non terminal symbol derives the replaced terminal symbol.
                    rule[item[1]] = f"{rule[0]}{str(index)}"
                    new_rules += [f"{rule[0]}{str(index)}", item[0]]
                index += 1
            # Rule is in form A -> X B C [...]
            while len(rule) > 3:
                new_rules.append([f"{rule[0]}{str(index)}", rule[1], rule[2]])
                rule = [rule[0]] + [f"{rule[0]}{str(index)}"] + rule[3:]
                index += 1
        # Adds the modified or unmodified (in case of A -> x i.e.) rules.
        # add_rule(rule)
        res_append(rule)
        if new_rules:
            cnf_grammar.extend(new_rules)
    # Handle the unit productions (A -> X)
    # while unit_productions:
    #     rule = unit_productions.pop()
    #     print(rule)
    #     if rule[1] in RULE_DICT:
    #         for item in RULE_DICT[rule[1]]:
    #             new_rule = [rule[0]] + item
    #             if len(new_rule) > 2 or new_rule[1][0] == "'":
    #                 cnf_grammar.insert(0, new_rule)
    #             else:
    #                 unit_productions.append(new_rule)
    #             add_rule(new_rule)
    return cnf_grammar


def main():
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("grammar")
    # # argparser.add_argument("sentence")
    # args = argparser.parse_args()                                   
    # grammar_file = args.grammar
    # sentence = args.sentence
    
    grammar_file = "l1.cfg"
    setence = "i called mom at home"
    gram = convert_grammar(read_grammar(grammar_file))
    print(gram)


# gm = data.load(grammar_file)
if __name__ == '__main__':
    main()