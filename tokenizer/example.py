#!/usr/bin/python3

from extract import raw_text
from simple_tokenizer import tokenizer

enc_text = tokenizer.encode(raw_text)

print(len(enc_text))
enc_type = enc_text[60:]
context_size = 4
x = enc_type[:context_size]
y = enc_type[1: context_size + 1]
print(f"x: {x}")
print(f"y:    {y}")

for i in range(1, context_size+1):
    context = enc_type[:i]
    desired = enc_type[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))