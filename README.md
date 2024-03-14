# LLM-AX650

## TinyLLaMa-INT8

```shell
# ./run_int8.sh "help me calculate sum 1-9 use c language:"
[I][                            Init][  71]: tokenizer init ok
[I][                            Init][  89]: embed_selector init ok
[I][                            Init][ 118]: init axmodel(tinyllama-int8-01/tinyllama_l0.axmodel) ok
...
[I][                            Init][ 118]: init axmodel(tinyllama-int8-01/tinyllama_l21.axmodel) ok
[I][                            Init][ 145]: init post axmodel(tinyllama-int8-01/tinyllama_post.axmodel) ok
[I][                            Init][ 151]: kv_cache_num: 1023

 advantage of using c language for this task is that we can use functions to calculate the sum of the first 10 natural numbers.

#include <stdio.h>

int main()
{
    int sum = 0;

    for (int I = 1; I <= 10; i++)
    {
        sum += i;
    }

    printf("sum of first 10 natural numbers is: %d\n", sum);

    return 0;
}


in this program, we first include the necessary header files for the program.

then, we declare a variable called `sum` to store the sum of the first 10 natural numbers.

next, we use a for loop to iterate through each natural number from 1 to 10.

in each iteration, we add the current number to the `sum` variable.

finally, we print the `sum` variable to the console.

this program uses the `for` loop to iterate through each natural number from 1 to 10.

for each number, we add it to the `sum` variable.

finally, we print the `sum` variable to the console.

this program uses the `printf` function to print a message to the console.

the `printf` function takes a format string and a variable number of arguments.

in this case, we are printing a message that includes the value of the `sum` variable.

in summary, this program uses the `for` loop to iterate through each natural number from 1 to 10.

for each number, we add it to the `sum` variable.

finally, we print the `sum` variable to the console
[N][                             Run][ 332]: hit eos
```

## TinyLLaMa-BF16
```shell
# ./run_bf16.sh "help me calculate sum 1-9 use c language:"
[I][                            Init][  71]: tokenizer init ok
[I][                            Init][  89]: embed_selector init ok
[I][                            Init][ 135]: read_file(tinyllama-bf16/tinyllama_l0.axmodel) ok
...
[I][                            Init][ 135]: read_file(tinyllama-bf16/tinyllama_l21.axmodel) ok
[I][                            Init][ 145]: init post axmodel(tinyllama-bf16/tinyllama_post.axmodel) ok
[I][                             Run][ 248]: kv_cache_num: 1023


#include <stdio.h>

int main() {
    int sum = 0;
    int num;

    printf("Enter a number: ");
    scanf("%d", &num);

    for (int I = 1; I <= num; i++) {
        sum += i;
    }

    printf("The sum of numbers from 1 to %d is: %d\n", num, sum);

    return 0;
}

this program prompts the user to enter a number and calculates the sum of numbers from 1 to the entered number using a for loop. The sum is then printed to the console.
[N][                             Run][ 332]: hit eos
```