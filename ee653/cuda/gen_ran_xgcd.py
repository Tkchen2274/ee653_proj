from random import randint

def generate_test_pairs(num_pairs: int, output_file: str = "input.txt") -> None:
    """
    Generate pairs of random 32-bit integers (0 to 2^32 - 1).
    
    Args:
        num_pairs: Number of test pairs to generate
        output_file: Output file path
    """
    MAX_32BIT = 2**32 - 1  # Maximum 32-bit unsigned integer
    
    with open(output_file, 'w') as f:
        for _ in range(num_pairs):
            a = randint(0, MAX_32BIT)
            b = randint(0, MAX_32BIT)
            f.write(f"{a} {b}\n")

if __name__ == "__main__":
    # Generate 1000 test pairs
    num_pairs = 1000
    print(f"Generating {num_pairs} pairs of 32-bit integers...")
    generate_test_pairs(num_pairs)
    print(f"Done! Test pairs written to input.txt")
