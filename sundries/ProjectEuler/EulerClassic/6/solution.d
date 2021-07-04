/*
 * Project Euler: Classic, "Sum square difference" solution
 * Implementation by A. S. "Aleksey" Ahmann <hackermaneia@riseup.net>
 * - https://github.com/Alekseyyy
 *
 * Problem link: https://projecteuler.net/problem=6
 */

import std.stdio : writeln;
import std.math.exponential : pow;
import std.conv : to;

void main() {
	float sum_of_squares = 0;
	float square_of_sums = 0;
	for (int i = 1; i <= 100; i++) {
		sum_of_squares = sum_of_squares + pow(to!float(i), 2);
		square_of_sums = square_of_sums + to!float(i);
	}
	square_of_sums = pow(to!float(square_of_sums), 2);
	writeln(to!int(square_of_sums - sum_of_squares));
}
