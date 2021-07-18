//#include <stdio.h>
//#include <iostream>

#include <stdio.h>
int main(void)
{
	int n, number, tri, counter;

	for (counter = 1; counter <= 5; ++counter) {
		printf("What tri number do you want? ");
		scanf_s("%i", &number);

		tri = 0;

		for (n = 1; n <= number; ++n)
			tri += n;

		printf("Tri number %i is %i\n\n", number, tri);
	}

	return 0;
}
