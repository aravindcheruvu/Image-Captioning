import json

import nltk
import pickle
import argparse
from collections import Counter, namedtuple


from FlickrJSON import FlickrJSON


class FlickrVocab(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(JSON_data, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    pythonObj = json.loads(JSON_data)
    print(len(pythonObj['annotations']))
    for i in range(len(pythonObj['annotations'])):
        caption = str(pythonObj['annotations'][i]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        print(caption)
    # # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    print("Length of Words: ",len(words))
    # Create a vocab wrapper and add some special tokens.
    vocab = FlickrVocab()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    i=0
    # # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):

    flickj = FlickrJSON()
    JSON_data = flickj.BuildJson(args.caption_path)
    vocab = build_vocab(JSON_data, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    # print("Test: {}".format(vocab("cat")))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='./Flickr8k_text/Flickr8k.token.txt',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./Flickr8k_text/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)