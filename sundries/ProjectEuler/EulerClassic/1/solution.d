/*
 * Project Euler: Classic, "Multiples of 3 and 5" solution
 * Implementation by A. S. "Aleksey" Ahmann <hackermaneia@riseup.net>
 * - https://github.com/Alekseyyy
 *
 * Problem link: https://projecteuler.net/problem=1
 */

import std.stdio : writeln;
import std.algorithm.iteration : sum;

void main() {
	int[] multiples;
	for (int i = 1; i <= 999; i++) {
		if (i % 3 == 0 || i % 5 == 0)
			multiples ~= i;
	}
	writeln(sum(multiples));
}
