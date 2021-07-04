#!/bin/python
#
# By A. S. "Aleksey" Ahmann (hackerrank.com/hackermaneia)
# https://github.com/Alekseyyy
#

import sys

def fibonacci(n):
    a, b, r = 1, 1, [1]
    while b < n:
        r.append(b)
        t_b = b
        b = b + a
        a = t_b
    return r

def iseven(x):
    if x % 2 == 0:
        return True
    return False

t = int(raw_input().strip())
inputs = []
for a0 in xrange(t):
    inputs.append(long(raw_input().strip()))
    
for n_g, k_g in enumerate(inputs):
    fl = fibonacci(k_g)
    evenfibs = [] 
    for n_l, k_l in enumerate(fl):
        if iseven(k_l):
            evenfibs.append(k_l)
    print sum(evenfibs)
