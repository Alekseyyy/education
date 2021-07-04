/*
 * Project Euler: Classic, "Even Fibonacci numbers" solution
 * Implementation by A. S. "Aleksey" Ahmann <hackermaneia@riseup.net>
 * - https://github.com/Alekseyyy
 *
 * Problem link: https://projecteuler.net/problem=2
 */

import std.stdio : writeln;
import std.algorithm.iteration : sum;

void main() {
	int[2] fibonacci = [1, 2];
	int[] even_fibonacci = [2];

	do {
		int fib_k = fibonacci[0] + fibonacci[1];
		if (fib_k % 2 == 0)
			even_fibonacci ~= fib_k;
		fibonacci[0] = fibonacci[1];
		fibonacci[1] = fib_k;
	} while (fibonacci[0] + fibonacci[1] <= 3999999);

	writeln(sum(even_fibonacci));
}
