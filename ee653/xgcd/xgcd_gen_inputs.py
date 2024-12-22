from random import getrandbits
import sympy

def generate_test_cases(num_cases: int, bits: int = 32, output_file: str = "input.txt") -> None:
    """
    Generate test cases with random 32-bit integers.
    
    Args:
        num_cases: Number of test cases to generate
        bits: Number of bits for the integers (default 32)
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        for _ in range(num_cases):
            # Generate two random integers
            a = getrandbits(bits)
            b = getrandbits(bits)
            
            # Write to file
            f.write(f"{a} {b}\n")

if __name__ == "__main__":
    print("Generating test cases with 32-bit integers...")
    generate_test_cases(1000)  # Generates 5 test cases
    print("Done! Test cases written to input.txt")
