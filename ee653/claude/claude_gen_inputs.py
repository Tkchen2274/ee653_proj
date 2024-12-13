from random import getrandbits
import math
from typing import Tuple
import sympy

def generate_safe_prime(bits: int) -> int:
    """
    Generate a safe prime (a prime p where (p-1)/2 is also prime).
    """
    while True:
        # Generate a prime candidate q
        q = sympy.randprime(2**(bits-2), 2**(bits-1))
        # Calculate potential safe prime p = 2q + 1
        p = 2 * q + 1
        # Check if p is prime
        if sympy.isprime(p):
            return p

def generate_vdf_parameters(bits: int = 256) -> Tuple[int, int, int, int]:
    """
    Generate parameters for VDF test cases.
    
    Args:
        bits: Number of bits for the parameters (default 256)
    
    Returns:
        Tuple of (x, N, b0, T)
        - x: challenge value
        - N: public modulus (product of safe primes)
        - b0: constant value
        - T: number of iterations
    """
    # Generate two safe primes for N
    p = generate_safe_prime(bits // 2)
    q = generate_safe_prime(bits // 2)
    N = p * q
    
    # Generate random challenge x
    x = getrandbits(bits) % N
    
    # Set b0 = p + q as suggested in the design
    b0 = p + q
    
    # Set T based on desired time delay
    # For testing, we'll use smaller values
    # In production, this should be much larger
    T = 1000
    
    return x, N, b0, T

def generate_test_cases(num_cases: int, output_file: str = "input.txt") -> None:
    """
    Generate multiple test cases and write them to a file.
    
    Args:
        num_cases: Number of test cases to generate
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        for _ in range(num_cases):
            # Generate parameters
            x, N, b0, T = generate_vdf_parameters()
            
            # Write to file in the required format
            f.write(f"{x} {N} {b0} {T}\n")

def generate_increasing_difficulty_cases(
    start_T: int = 100, 
    max_T: int = 10000, 
    num_cases: int = 5, 
    output_file: str = "input.txt"
) -> None:
    """
    Generate test cases with increasing difficulty (T values).
    
    Args:
        start_T: Starting value for T
        max_T: Maximum value for T
        num_cases: Number of test cases to generate
        output_file: Path to output file
    """
    # Calculate T values on a logarithmic scale
    T_values = [
        int(start_T * math.exp(i * math.log(max_T/start_T) / (num_cases-1)))
        for i in range(num_cases)
    ]
    
    with open(output_file, 'w') as f:
        for T in T_values:
            # Generate parameters
            x, N, b0, _ = generate_vdf_parameters()
            
            # Write to file with current T value
            f.write(f"{x} {N} {b0} {T}\n")

if __name__ == "__main__":
    print("Generating standard test cases...")
    generate_test_cases(3)
    
    print("Generating difficulty progression test cases...")
    generate_increasing_difficulty_cases(
        start_T=100,
        max_T=10000,
        num_cases=5,
        output_file="input_difficulty.txt"
    )
    
    print("Test case generation complete!")
