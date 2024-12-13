gcc -o gpt256 gpt_serial.c -lgmp
gcc -o verify verify.c -lgmp
./gpt256
./verify
