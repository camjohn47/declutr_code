import sys
from os import getcwd
from os.path import dirname, join, exists
from pickle import load

MAIN_DIR = dirname(getcwd())
TOKENIZER_SUBDIR = "tokenizers"
TOKENIZER_DIR = join(MAIN_DIR, TOKENIZER_SUBDIR)

_, tokenizer_id = sys.argv
tokenizer_path = join(TOKENIZER_DIR, f"{tokenizer_id}.pickle")

if not exists(tokenizer_path):
    raise FileNotFoundError(f"ERROR: Tokenizer path = {tokenizer_path} doesn't exist.")

tokenizer_file = open(tokenizer_path, "rb")
tokenizer = load(tokenizer_file)
print(f"UPDATE: Tokenizer config  = {tokenizer.get_config()}")

