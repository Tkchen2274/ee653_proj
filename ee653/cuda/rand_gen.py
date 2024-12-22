import struct
import random

def int_to_32bit_digits(number):
    """Convert a large integer into an array of 8 32-bit digits"""
    digits = []
    for _ in range(8):  # 8 digits for 256 bits
        digits.append(number & 0xFFFFFFFF)
        number >>= 32
    return digits

def write_binary_file(filename, number):
    """Write the 32-bit digits to a binary file"""
    digits = int_to_32bit_digits(number)
    with open(filename, 'wb') as f:
        for digit in digits:
            f.write(struct.pack('<I', digit))  # little-endian 32-bit unsigned int

def generate_prime_like_number(bits):
    """Generate a large number that's suitable for GCD testing"""
    # Generate a random odd number of specified bits
    number = random.getrandbits(bits)
    # Make sure it's odd (better for GCD)
    return number | 1

def main():
    # Generate two large numbers (around 256 bits each)
    # Using slightly smaller numbers to ensure no overflow
    a = generate_prime_like_number(250)
    b = generate_prime_like_number(250)
    
    # Make sure a > b
    if b > a:
        a, b = b, a
    
    print(f"Generated numbers:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a (hex) = {hex(a)}")
    print(f"b (hex) = {hex(b)}")
    
    # Write to binary files
    write_binary_file('a_input.bin', a)
    write_binary_file('b_input.bin', b)
    
    print("\nBinary files 'a_input.bin' and 'b_input.bin' have been created.")
    print("You can now use these files with your CUDA GCD program.")

if __name__ == "__main__":
    main()
