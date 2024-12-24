#!/usr/bin/python3
import re

with open("verdict.txt", "r", encoding="utf-8") as f:
   raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(len(preprocessed))

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
#print(vocab_size)

voca = {token:integer for integer,token in enumerate(all_words)}
