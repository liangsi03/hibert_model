import os
import pandas as pd
import nltk
import json
import sys
from nltk.tokenize import RegexpTokenizer

texts = []
word_dict = {}
id_dict = {}
directory = '../author_content_summary'
tokenizer = RegexpTokenizer(r'\w+')

def read_data():
    for filename in os.listdir(directory):
        with open(directory + '/' + filename, 'r') as read_file:
            data = json.load(read_file)
            for i in range(0,len(data)):
                content = data[i][1]
                summary = data[i][2]
                texts.append(content)
                texts.append(summary)    
    return texts        



def gen_vocab(texts):
    word_list = []
    for text in texts:
        text = text.strip().replace("/", " and ").replace("-", " ")
        for word in tokenizer.tokenize(text):
            word = word.replace("..", "")
            if not word.isdigit():
                word_list.append(word)
    word_list = list(set(word_list))
    # We need to tell LSTM the start and the end of a sentence.
    # And to deal with input sentences with variable lengths,
    # we also need padding position as 0.
    # You can see more details in the latter part.
    word_list = ["_PAD_", "_BOS_", "_EOS_", "_BOP_", "_NUM_", "_UNK_"] + word_list
    with open("data/vocab", "w") as vocab_file:
        for word in word_list:
            vocab_file.write(word + "\n")  
    return len(word_list)



def gen_id_seqs():

    def word_to_id(word, word_dict):
        id = word_dict.get(word)
        return id if id is not None else word_dict.get("_UNK_")

    def text_to_id(text):
        sentences = nltk.sent_tokenize("_BOP_ " + text.replace("\n", "_BOP_"))
        seqs = []
        for sentence in sentences:
            sentence = sentence.replace("/", " and ").replace("-", " ")
            words = tokenizer.tokenize(sentence)
            seq = []
            while (words != [] and words[0] == "_BOP_"):
                if len(seq) == 0:
                    seq.append(word_to_id(words[0], word_dict))
                words.pop(0)
            seq.append(1)
            for i in range(len(words)):
                word = words[i].replace("..", "")
                if word.isdigit():
                    seq.append(word_to_id("_NUM_", word_dict))                    
                elif not (word == "_BOP_" and words[i-1] == "_BOP_"):
                    seq.append(word_to_id(word, word_dict))
            seq.append(2)
            seqs.append(seq)
        return seqs

    with open("data/vocab", "r") as vocab_file:
        lines = [line.strip() for line in vocab_file.readlines()]
        word_dict = dict([(b, a) for (a, b) in enumerate(lines)])

    for filename in os.listdir(directory):
        texts = []
        with open(directory + '/' + filename, 'r') as read_file:
            data = json.load(read_file)
            for i in range(0,len(data)):
                content = text_to_id(data[i][1])
                summary = text_to_id(data[i][2])
                texts.append(content)
                texts.append(summary)    
        with open('data/' + filename, 'w') as f:
            json.dump(texts, f)
            print("encoded " + filename)



def main():
    print("start working...")
    texts = read_data()
    print("data read.")
    vocab_size = gen_vocab(texts)
    print("vocab size is: " + str(vocab_size))
    gen_id_seqs()
    print("tokens encoded.")

main()