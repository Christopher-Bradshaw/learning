/* Program to find unmatched parens/brackets/quotes.
 * Does not handle comments. Does handle \ escaping in strings.
 * Mostly for me to relearn and comment on the basics of C. */

#include <stdio.h> // Includes use < when they come from the stdlib
#include <stdlib.h>
#include <string.h>

#define OPENING_BRACES "({["
#define CLOSING_BRACES ")}]"

#define STRING_BRACES "\"'"

int read_until_end_of_string(char string_char);

// External (global) variable, if you want to use this in a func, re-declare with extern.
// You probably do not want to use external vars but I put it here to demonstrate.
int line_no = 1;

int main() {

    extern int line_no;
    long index;

    int c;
    char *loc, *brace_stack = malloc(100 * sizeof(char)); // We should keep track of what we have used and realloc if needed
    char tmp_str[2] = "\0\0";

    while ((c = getchar()) != EOF) { // Assignment is an expression so you can compare the result.

        if (strchr(STRING_BRACES, c) != NULL) {
            if (read_until_end_of_string(c) == -1) {
                return 1;
            }
        }
        else if (strchr(OPENING_BRACES, c) != NULL) {
            tmp_str[0] = (char)c;
            brace_stack = strcat(brace_stack, tmp_str);
        }
        else if ((loc = strchr(CLOSING_BRACES, c)) != NULL) {
            index = (long)(loc) - (long)(CLOSING_BRACES);
            if (brace_stack[strlen(brace_stack) - 1] == OPENING_BRACES[index]) {
                brace_stack[strlen(brace_stack) - 1] = '\0';
            }
            else {
                printf("Unmatched closing brace %c on line %d\n", (char)c, line_no);
                return 1;
            }
        }
        else if (c == '\n') {
            line_no++;
        }
    }
    if (strlen(brace_stack) != 0) {
        printf("Unclosed braces!\n");
        return 1;
    }
    printf("No errors in %d lines\n", line_no);
    return 0;
}

int read_until_end_of_string(char string_char) {
    extern int line_no;
    int c, str_start = line_no;
    while ((c = getchar())) {
        if (c == EOF) {
            printf("EOF while in string starting on line %d\n", str_start);
            break;
        }
        // Ignore escaped chars in strings
        if (c == '\\') {
            getchar();
        }
        else if (c == '\n') {
            line_no++;
        }
        else if (c == string_char) {
            return 0;
        }
    }
    return -1;
}
