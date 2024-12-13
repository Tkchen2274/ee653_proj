# python claude_gen_inputs.py
gcc -o claude256 claude_serial.c -lgmp
gcc -o claudeverify claude_verify.c -lgmp
./claude256
./claudeverify
