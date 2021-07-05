/*
 * Project Euler: Classic, "Special Pythagorean triplet" solution
 * Implementation by A. S. "Aleksey" Ahmann <hackermaneia@riseup.net>
 * - https://github.com/Alekseyyy
 *
 * Problem link: https://projecteuler.net/problem=9
 */

import std.stdio : writeln;
import std.conv : to;
import std.math.exponential : pow;

void main() {
	int a;
	int b;
	int c;
	// I used a brute force solution cos' I'm feeling lazy today :P
	for (int x = 1; x <= 1000; x++) {
		for (int y = 1; y <= 1000; y++) {
			for (int z = 1; z <= 1000; z++) {
				if (criteria(x, y, z)) {
					a = x;
					b = y;
					c = z;
				}
			}
		}
	}
	writeln(a * b * c);
}

bool criteria(int a, int b, int c) {

	if (a < b && b < c) {
		if (a + b + c == 1000) {
			if (a * a + b * b == c * c)
				return true;
		}
	}
	return false;
}

