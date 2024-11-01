def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

def reverse_string(s):
    return s[::-1]

def is_even(n):
    return n % 2 == 0

def is_odd(n):
    return n % 2 != 0

def find_max(lst):
    return max(lst)

def find_min(lst):
    return min(lst)

def sum_list(lst):
    return sum(lst)

def avg_list(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

def count_occurrences(lst, value):
    return lst.count(value)

def square(n):
    return n ** 2

def cube(n):
    return n ** 3

def power(base, exp):
    return base ** exp

def sort_list(lst):
    return sorted(lst)

def remove_duplicates(lst):
    return list(set(lst))

def capitalize_string(s):
    return s.capitalize()

def title_string(s):
    return s.title()

def is_palindrome(s):
    return s == s[::-1]

def flatten_list(lst):
    return [item for sublist in lst for item in sublist]








