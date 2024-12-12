import random

def generate_random_numbers(filename, num_lines, min_value, max_value):
    """
    Generate a text file with two numbers on each line.
    The pairs follow the rules: odd & odd, even & odd, or odd & even (no even & even pairs).

    :param filename: Name of the output text file.
    :param num_lines: Number of lines to generate.
    :param min_value: Minimum value for the random numbers.
    :param max_value: Maximum value for the random numbers.
    """
    with open(filename, 'w') as file:
        for _ in range(num_lines):
            while True:
                num1 = random.randint(min_value, max_value)
                num2 = random.randint(min_value, max_value)
                
                # Check conditions: avoid even & even
                if not (num1 % 2 == 0 and num2 % 2 == 0):
                    file.write(f"{num1} {num2}\n")
                    break

# Parameters
output_filename = "input.txt"  # Name of the output file
lines_to_generate = 2000                 # Number of lines to generate
min_random_value = 1                    # Minimum random number value
max_random_value = 4294967295                  # Maximum random number value

# Generate the file
generate_random_numbers(output_filename, lines_to_generate, min_random_value, max_random_value)
print(f"File '{output_filename}' has been generated with {lines_to_generate} lines.")

