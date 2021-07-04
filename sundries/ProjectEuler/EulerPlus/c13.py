#!/bin/python
#
# By A. S. "Aleksey" Ahmann (hackerrank.com/hackermaneia)
# https://github.com/Alekseyyy
#
# Enter your code here. Read input from STDIN. Print output to STDOUT

T = int(raw_input().strip())

to_sum = []
for i in range(T):
    to_sum.append(int(raw_input().strip()))
    
final_sum = sum(to_sum)
final_sum = str(final_sum)
print final_sum[0:10]
