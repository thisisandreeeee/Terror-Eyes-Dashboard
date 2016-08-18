"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys
import json
import pandas as pd
import networkx as nx
import numpy as np
import csv

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser

def make_csv_for_apriori():
    df = pd.read_csv('csv-files/gtd_2011to2014.csv', encoding='Latin-1',low_memory=False)
    df = df[df['multiple'] == '1']
    col = 'targtype1'
    df[col] = df[col].apply(pd.to_numeric, args=('coerce',))
    df = make_network(df) # adds another column called marker that tells us when a bunch of related attacks are well, related. we need to do this because the related column isn't able to act as an id cos it's a bitch
    with open('csv-files/apriori.csv', 'w') as f:
        w = csv.writer(f)
        for name, group in df.groupby('marker'):
            row = list(set([i for i in group[col]]))
            w.writerow(row)

def make_network(df):
    '''
    adds all the related points to a undirected graph, then finds all the connected components and gives them an id
    '''
    G = nx.Graph()
    for i in df.index:
        _from = df['eventid'][i]
        _to = df['related'][i]
        if type(_to) == str:
            _to = _to.split(",")
            for t in _to:
                G.add_edge(_from, t.strip())
    marker = pd.Series(index = df.index)
    cc = [(i, comp) for i, comp in enumerate(list(nx.connected_components(G)))]
    for i in df.index:
        eventid = df['eventid'][i]
        for ind, comp in cc:
            if eventid in comp:
                marker[i] = ind

    df['marker'] = marker
    return df


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
                for transaction in transactionList:
                        if item.issubset(transaction):
                                freqSet[item] += 1
                                localSet[item] += 1

        for item, count in localSet.items():
                support = float(count)/len(transactionList)

                if support >= minSupport:
                        _itemSet.add(item)

        return _itemSet


def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data_iter, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport,
                                        freqSet)

    currentLSet = oneCSet
    k = 2
    while(currentLSet != set([])):
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet,
                                                transactionList,
                                                minSupport,
                                                freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
            """local function which Returns the support of an item"""
            return float(freqSet[item])/len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])

    toRetRules = []
    # for key, value in largeSet.items()[1:]:
    count = 0
    for key, value in largeSet.items():
        if count == 0:
            count += 1
            continue
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item)/getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)),
                                           confidence))
    return toRetItems, toRetRules


def printResults(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    d = {}
    for rule in rules:
        rs, conf = rule
        cond, res = rs
        if len(cond) == 1:
            d[cond[0]] = [i for i in res]
    print(items, "\n\n##########\n\n", rules)
    open("dics/apriori.json",'w').write(json.dumps(d, indent=4))
    # for item, support in sorted(items, key=lambda (item, support): support):
    #     print "item: %s , %.3f" % (str(item), support)
    # print "\n------------------------ RULES:"
    # for rule, confidence in sorted(rules, key=lambda (rule, confidence): confidence):
    #     pre, post = rule
    #     print "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)


def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = line.strip().rstrip(',')                         # Remove trailing comma
                record = frozenset(line.split(','))
                yield record


if __name__ == "__main__":
    make_csv_for_apriori()
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default=None)
    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value',
                         default=0.15,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='minC',
                         help='minimum confidence value',
                         default=0.6,
                         type='float')

    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
            inFile = sys.stdin
    elif options.input is not None:
            inFile = dataFromFile(options.input)
    else:
            print('No dataset filename specified, system with exit\n')
            sys.exit('System will exit')

    minSupport = options.minS
    minConfidence = options.minC

    items, rules = runApriori(inFile, minSupport, minConfidence)

    printResults(items, rules)
