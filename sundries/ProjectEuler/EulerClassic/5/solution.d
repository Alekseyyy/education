/*
 * Project Euler: Classic, "Smallest multiple" solution
 * Implementation by A. S. "Aleksey" Ahmann <hackermaneia@riseup.net>
 * - https://github.com/Alekseyyy
 *
 * Problem link: https://projecteuler.net/problem=5
 */

import std.stdio : writeln;

bool criteria(int n) {
	for (int i = 1; i <= 20; i++) {
		if (n % i != 0)
			return false;
	}
	return true;
}

void main() {
	int test = 0;
	bool done = false;
	do {
		test = test + 1;
		if (criteria(test)) {
			writeln(test);
			done = true;
		}
	} while (!done);
}

