/*
 * Project Euler: Classic, "Summation of primes" solution
 * Implementation by A. S. "Aleksey" Ahmann <hackermaneia@riseup.net>
 * - https://github.com/Alekseyyy
 *
 * Problem link: https://projecteuler.net/problem=10
 * I ported some code from here: https://www.geeksforgeeks.org/python-program-for-sieve-of-eratosthenes/
 */

import std.stdio : writeln, printf;
import std.algorithm.mutation : fill;
import std.algorithm.iteration : sum;

long[] sieve(long n) {

	//printf("On sieve function with n = %i\n", n);
	bool[] bsieve = new bool[n + 1];
	fill(bsieve, true);

	for (long p = 2; p * p <= bsieve.length - 1; p++) {
		if (bsieve[p] == true) {
			for (long i = p * p; i <= bsieve.length - 1; i += p)
				bsieve[i] = false;
		}
	}

	long[] result;
	for (long p = 2; p <= bsieve.length - 1; p++) {
		if (bsieve[p])
			result ~= p;
	}
	return result;
}

void main() {

	long[] primes = sieve(1999999);
	writeln(sum(primes));
}
