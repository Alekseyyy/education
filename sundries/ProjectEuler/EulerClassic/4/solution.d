/*
 * Project Euler: Classic, "Largest palindrome product" solution
 * Implementation by A. S. "Aleksey" Ahmann <hackermaneia@riseup.net>
 * - https://github.com/Alekseyyy
 *
 * Problem link: https://projecteuler.net/problem=4
 */

import std.stdio : writeln;
import std.conv : to;
import std.algorithm : reverse;

bool isPalindrome(int number) {
	return to!string(number) == to!string(number).dup.reverse;
}

void main() {
	int winner = 0;
	for (int i = 100; i <= 999; i++) {
		for (int j = 100; j <= 999; j++) {
			if (isPalindrome(i * j) && i * j > winner)
				winner = i * j;
		}
	}
	writeln(winner);
}
