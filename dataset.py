import torch as tr
import torch.utils.data as data
import random

"""

Code implementation based on demoed code in CIS 700: Deep Automated Theorem Proving

Title: TransformerHolstep.ipynb
Author: Dr. Garrett Katz
Date: Spring 2023

"""


"""

Embedder

"""

def sanitize(token):
    return token.replace(".", "_")

class Embedder(tr.nn.Module):
    def __init__(self, vocab, cutoff_freq, d_model):
        super(Embedder, self).__init__()

        self.embeddings = tr.nn.ParameterDict()
        for token, frequency in vocab.items():
            if frequency >= cutoff_freq:
                self.embeddings[sanitize(token)] = tr.nn.Parameter(tr.randn(d_model) / d_model**.5)

        self.unknown_embedding = tr.nn.Parameter(tr.randn(d_model) / d_model**.5)


    def forward(self, tokens):
        seq = []
        for token in tokens:
            seq.append( self.embeddings.get(sanitize(token), self.unknown_embedding) )
        return tr.stack(seq).unsqueeze(1) # insert singleton dimension for batch


"""

Encoder

"""


# Convert to single expression encoder
def encode(premise, max_len, embedder):

    # ignore leading example tokens if conjecture+example is longer than max length
    context_length = max_len - len(premise)
    prompt = premise[-context_length:]

    # embed the prompt tokens with position information
    encoded = embedder(prompt).permute(1, 0, 2)
    encoded = tr.nn.functional.pad(encoded, (0, 0, 0, max_len-encoded.size()[1], 0, 0), mode='constant', value=0)

    return encoded


"""

Parsing

"""

def parse_tokens(line):
    # whitespace
    line = line[2:].rstrip()
    tokens = []
    token = ""
    for c in line:
        if c == " ":
            if token != "":
                tokens.append(token)
                token = ""
        elif c in "()":
            if token != "":
                tokens.append(token)
            tokens.append(c)
            token = ""
        else:
            token += c

    # don't forget last token
    if token != "":
        tokens.append(token)

    return tokens
        
def get_samples():
    for file_num in range(9999):
        with open(f"data/holstep/train/{file_num+1:05d}", "r") as f:

            # conjecture block
            f.readline() # name
            f.readline() # description
            line = f.readline() # tokenization
            conjecture = parse_tokens(line)

            # remaining blocks
            dependencies = []
            examples, labels = [], []
            while True:

                line = f.readline()
                if line == "": break # file done

                if line[0] == "D": # dependency block
                    f.readline() # description
                    line = f.readline()
                    dependency = parse_tokens(line) # token list
                    dependencies.append(dependency)

                if line[0] in "+-": # example
                    label = line[0]
                    example = parse_tokens(line)# token list
                    examples.append(example)
                    labels.append(label)

        yield conjecture, dependencies, examples, labels



"""

Dataset Builder

"""

def dataset_builder(size):

    # Dictionaries for making histograms: dict[key] = count of key in dataset
    vocab = {}
    prompt_lengths = {}
    max_len = 256
    train_split = 0.7
    test_split = 0.1

    for s, (con, dep, examples, labels) in enumerate(get_samples()):

        for token in con: vocab[token] = vocab.get(token, 0) + 1
        plen = len(con)

        for example, label in zip(examples, labels):
            for token in example: vocab[token] = vocab.get(token, 0) + 1

            plen_e = plen + len(example)
            prompt_lengths[plen_e] = prompt_lengths.get(plen_e, 0) + 1

        if s == size: break

    embedder = Embedder(vocab, size, max_len)

    dset = []
    for update, (con, dep, examples, labels) in enumerate(get_samples()):
        
        if len(dset) >= size: break
        
        for idx in range(len(examples)):

            if labels[idx] == '-': label = tr.tensor([0., 1.])
            else: label = tr.tensor([1., 0.])

            seq = (con, examples[idx], label)
            dset.append(seq)

    dset = dset[0:size]
    random.shuffle(dset)

    # Calculate number of samples
    num_data = len(dset)
    num_train = int(train_split * num_data)
    num_test = num_train + int(test_split * num_data)

    # Define samplers for training and testing subsets
    train_sampler = dset[0:num_train]
    val_sampler = dset[num_train:num_test]
    test_sampler = dset[num_test:]

    return train_sampler, val_sampler, test_sampler, embedder