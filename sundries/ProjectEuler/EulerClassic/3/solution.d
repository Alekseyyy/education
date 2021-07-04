/*
 * Project Euler: Classic, "Largest prime factor" solution
 * Implementation by A. S. "Aleksey" Ahmann <hackermaneia@riseup.net>
 * - https://github.com/Alekseyyy
 *
 * Problem link: https://projecteuler.net/problem=3
 * I ported some code from here: https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
 */

import std.stdio : write, writeln;
import std.conv : to;
import core.math : sqrt;

void main() {
	long n = 600851475143;
	long[] extracted_primes;

	while (n % 2 == 0) {
		//write(to!string(2), " ");
		n = n / 2;
	}
	for (long i = 3; i <= to!long(sqrt(to!float(n))); i = i + 2) {
		while (n % i == 0) {
			//write(to!string(2), " ");
			n = n / i;
		}
		if (n > 2)
			extracted_primes ~= n;
	}

	writeln(extracted_primes[extracted_primes.length - 1]);
}
