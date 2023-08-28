import sys

if len(sys.argv) < 2:
    print("Usage: generate.py [language]")
    sys.exit()

else:
    language = sys.argv[1]
    start_letters = sys.argv[2]

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataloader import *
from model import *

rnn = torch.load('conditional-char-rnn.pt')

# Generating from the Network
max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A', temperature=0.5):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)

            # Sample as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            topi = torch.multinomial(output_dist, 1)[0]

################## The method below can also be used to get top probabilities ##############
            # topv, topi = output.topk(1)
            # topi = topi[0][0]
############################################################################################

            # Stop at EOS, or add to output_str
            if topi == EOS:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


samples(language, start_letters=start_letters)