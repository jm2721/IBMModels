''' Juan Marron
    Some sentence parsing capabilities borrowed from the following github repository:
        https://github.com/alopez/en600.468
'''

#!/usr/bin/env python
import optparse
import sys
import math
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

''' f_count and e_count are dictionaries where words are keys and their frequencies are values (initialized to zero)
    fe_count is a dictionary where keys are tuples of words in english and french and values are the frequency in which
    each particular combination appears.
    Bitext is an array of arrays, where each inner array contains two elements corresponding to the same sentence in english
    and french
'''

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)

sys.stderr.write("Training with IBM Model 1...")
sys.stderr.write("\n")

''' Get alignments using IBM model 1 '''
# Implement IBM Model 1

# Counts initialize to zero.
fe_count = defaultdict(int)
#count = defaultdict(int)
total = defaultdict(int)
s_total = defaultdict(int)
t_fe = defaultdict(int)
iterations = 0

# number of unique words in the corpus. Used to initialize t(f|e) uniformly
num_words = 0
for sentence_pair in bitext:
    for word in set(sentence_pair[1]):
        num_words+=1
sys.stderr.write("Counting number of words \n")

for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
        for e_j in set(e):
            # Initialize uniformly
            t_fe[(f_i,e_j)] = 1.0/num_words
sys.stderr.write("Initializing t(f|e) uniformly\n")

sys.stderr.write("Starting iterations\n")
while iterations < 5:
    sys.stderr.write("Iteration: " + str(iterations) + "\n")
    #initialize all word pairs to 0
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            total[f_i] = 0
            for e_j in set(e):
                fe_count[(f_i,e_j)] = 0
    for (n, (f, e)) in enumerate(bitext):
        # For all *unique* words f_i in f
        for e_j in set(e):
            #f_count[f_i] += 1
            #n_f = f_count[f_i]
            s_total[e_j] = 0
            for f_i in set(f):
                s_total[e_j] += t_fe[(f_i, e_j)]
        for e_j in set(e):
            for f_i in f:
                fe_count[(f_i, e_j)] += t_fe[(f_i, e_j)]/s_total[e_j]
                total[f_i] += t_fe[(f_i, e_j)]/s_total[e_j]
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            for e_j in set(e):
                t_fe[(f_i,e_j)] = fe_count[(f_i, e_j)]/total[f_i]
    iterations+=1

#Decoding with IBM Model 1
# Implementation of IBM Model 2 with reparametrization
a = defaultdict(int)
count_a = defaultdict(int)
total_a = defaultdict(int)

sys.stderr.write("Training with IBM Model 2...\n")
#values of t(f|e) carried over from before
#initialize a(i|j, l_e, l_f) = 1/(l_f + 1) for all i, j, l_e, l_f

for (n, (f, e)) in enumerate(bitext):
    for (j, e_j) in enumerate(e):
        for (i, f_i) in enumerate(f):
            # Initialize uniformly
            a[(i, j, len(e), len(f))] = 1.0/(len(f) + 1)

iterations = 0
while iterations < 3:
    sys.stderr.write("Iteration: " + str(iterations) + "\n")
    for (n, (f, e)) in enumerate(bitext):
        for (i, f_i) in enumerate(set(f)):
            total[f_i] = 0
            for (j, e_j) in enumerate(set(e)):
                fe_count[(f_i,e_j)] = 0
                count_a[(i, j, len(e), len(f))] = 0
                total_a[(j, len(e), len(f))] = 0
    
    for (n, (f, e)) in enumerate(bitext):
        l_e = len(e)
        l_f = len(f)
        for (j, e_j) in enumerate(e):
            s_total[e_j] = 0
            for (i, f_i) in enumerate(f):
                s_total[e_j] += t_fe[(f_i, e_j)] * a[(i, j, len(e), len(f))]
                #sys.stderr.write(str(t_fe[(e_j, f_i)]) + ":" + str(a[(i, j, len(e), len(f))]) + '\n')
        for (j, e_j) in enumerate(e):
            for (i, f_i) in enumerate(f):
                c = t_fe[(f_i, e_j)] * a[(i, j, l_e, l_f)] / s_total[e_j]
                fe_count[(f_i, e_j)] += c
                total[f_i] += c
                count_a[(i, j, l_e, l_f)] += c
                total_a[(j, l_e, l_f)] += c
        
    # Estimate probabilities
    for (n, (f, e)) in enumerate(bitext):
        for f_i in enumerate(set(f)):
            for e_j in enumerate(set(e)):
                t_fe[(f_i, e_j)] = 0
                a[(i, j, len(e), len(f))] = 0
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            for e_j in set(e):
                t_fe[(f_i,e_j)] = fe_count[(f_i, e_j)]/total[f_i]
    for (i, j, l_e, l_f) in a:
        if (total_a[(j, l_e, l_f)] != 0):
            # Tried to implement model 2 reparametrization here
            a[(i, j, l_e, l_f)] = count_a[(i, j, l_e, l_f)] * math.exp(-abs((i+1)/l_f - (j+1)/l_e)) / total_a[(j, l_e, l_f)]
    iterations+=1

for (f, e) in bitext:
    for (i, f_i) in enumerate(f): 
        # Best alignment and the corresponding position
        best_prob = 0
        best_j = 0
        for (j, e_j) in enumerate(e):
            if a[(i, j, len(e), len(f))] * t_fe[(f_i,e_j)] > best_prob:
                #if (abs(i-j)) < 3:
                best_prob = a[(i, j, len(e), len(f))] * t_fe[(f_i,e_j)]
                best_j = j
                #else:
                    
        sys.stdout.write("%i-%i " % (i,best_j))
    sys.stdout.write("\n")
