#!/bin/python
#
# By A. S. "Aleksey" Ahmann (hackerrank.com/hackermaneia)
# https://github.com/Alekseyyy
#

import sys

def ss_1(k):
    sum_of_squares, square_of_sums = [], 0
    
    for i in range(1, k + 1):
        sum_of_squares.append(i)
        square_of_sums = square_of_sums + i**2
    
    sum_of_squares = sum(sum_of_squares)**2
    return sum_of_squares - square_of_sums

t = int(raw_input().strip())
for a0 in xrange(t):
    n = int(raw_input().strip())
    print ss_1(n)

