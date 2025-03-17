import os
import sys
import random
import itertools
import torch
from torch.utils.data import Dataset
root_dir = os.path.abspath(os.path.join(os.getcwd())) 
sys.path.insert(0, root_dir)
from config import SEQ_LEN

class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=SEQ_LEN):
        self.lines = data_pair
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        s1, s2, label = self.create_sentence_pair(idx)
        s1_output, s1_label = self.random_word(s1)
        s2_output, s2_label = self.random_word(s2)

        s1 = [self.tokenizer.vocab['[CLS]']] + s1_output + [self.tokenizer.vocab['[SEP]']]
        s1_label =  [self.tokenizer.vocab['[PAD]']] + s1_label + [self.tokenizer.vocab['[PAD]']]
        s2 = s2_output + [self.tokenizer.vocab['[SEP]']]
        s2_label = s2_label + [self.tokenizer.vocab['[PAD]']]

        bert_input = (s1 + s2)[:self.seq_len]
        bert_label = (s1_label + s2_label)[:self.seq_len]
        segment_label = [1] * len(s1) + [2] * len(s2)
        segment_label = segment_label[:self.seq_len]

        padding_length = self.seq_len - len(bert_input)
        bert_input += [self.tokenizer.vocab['[PAD]']] * padding_length
        bert_label += [self.tokenizer.vocab['[PAD]']] * padding_length
        segment_label += [self.tokenizer.vocab['[PAD]']] * padding_length
        output = {
            "bert_input": torch.tensor(bert_input),
            "bert_label": torch.tensor(bert_label),
            "segment_label": torch.tensor(segment_label),
            "is_consecutive": torch.tensor(label),
        }
        return output


    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []
        # from BERT paper: we mask 15% of all WordPiece tokens in each sequence at random
        #                  We only predict the masked words, rather than reconstructing the entire input
        for i, token in enumerate(tokens):
            probability = random.random()
            # Removing [CLS] and [SEP] tokens
            token_id = self.tokenizer(token)['input_ids'][1:-1]
            if probability < 0.15:
                probability /= 0.15
                if probability < 0.8:
                    # replace ith token with [MASK] 80% of the time 
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])  
                elif probability < 0.9:
                    # replace ith token with random token 10% of the time
                    for i in range(len(token_id)):
                        output.append(random.randint(0, len(self.tokenizer.vocab)-1))
                else:
                    # keep the ith token unchanged 10% of the time
                    output.append(token_id)
                output_label.append(token_id)
            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)
        # flatten output lists:
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        return output, output_label
    

    def get_sentence_pair(self, idx):
        # Choosing a sentence pair with a given index
        # Sentence2 is a contextual follow-up to sentence1
        sentence1 = self.lines[idx][0]
        sentence2 = self.lines[idx][1]
        return sentence1, sentence2


    def get_random_line(self):
        # Choosing a second sentence from a randomly picked sentence pair
        random_line_idx = random.randint(0, len(self.lines)-1)
        return self.lines[random_line_idx][1]
    

    def create_sentence_pair(self, idx):
        """
        Creates a sentence pair with a label indicating if the second sentence is a follow-up to the first.

        Two cases:
        - Probability > 0.5:
            1. sentence2 is a follow-up to sentence1
        - Else:
            2. sentence2 is a randomly picked line from get_random_line

        Args:
            idx (int): The index of the sentence pair to retrieve.

        Returns:
            tuple: A tuple containing:
                - s1 (str): The first sentence.
                - s2 (str): The second sentence.
                - label (int): A label indicating if s1 and s2 are a follow-up pair (1) or not (0).
        """
        s1, s2 = self.get_sentence_pair(idx)
        probability = random.random()
        # print(probability)
        if probability > 0.5:
            return s1, s2, 1
        else:
            incorrect_s2 = self.get_random_line()
            return s1, incorrect_s2, 0