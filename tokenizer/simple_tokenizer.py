#!/usr/bin/python3
import re
from extract import voca

class SimpleTokenizerV1:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text): 
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int 
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
 
    def decode(self, ids): 
        text = " ".join([self.int_to_str[i] for i in ids]) 
        
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) 
        return text

tokenizer = SimpleTokenizerV1(voca)

if __name__ == "__main__":
    tokenizer = SimpleTokenizerV1(voca)
    text = """"It's the last he painted, you know," 
    Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)