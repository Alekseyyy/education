/*
 * Project Euler: Classic, "10001st prime" solution
 * Implementation by A. S. "Aleksey" Ahmann <hackermaneia@riseup.net>
 * - https://github.com/Alekseyyy
 *
 * Problem link: https://projecteuler.net/problem=7
 * I ported some code from here: https://www.geeksforgeeks.org/python-program-for-sieve-of-eratosthenes/
 */

import std.stdio : writeln, printf;
import std.algorithm.mutation : fill;

int[] sieve(int n) {

	//printf("On sieve function with n = %i\n", n);
	bool[] bsieve = new bool[n + 1];
	fill(bsieve, true);

	for (int p = 2; p * p <= bsieve.length - 1; p++) {
		if (bsieve[p] == true) {
			for (int i = p * p; i <= bsieve.length - 1; i += p)
				bsieve[i] = false;
		}
	}

	int[] result;
	for (int p = 2; p <= bsieve.length - 1; p++) {
		if (bsieve[p])
			result ~= p;
	}
	return result;
}

void main() {

	int i = 0;
	int[] primes;
	do {
		primes = sieve(i);
		i = i + 10000;
	} while (primes.length <= 10001);
	writeln(primes[10000]);
}
