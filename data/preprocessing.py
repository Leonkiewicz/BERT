
import os
import tqdm
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer


def prepare_qa_pairs(lines, conv, MAX_LEN=64):
    lines_dict = {}
    i = 0
    for line in lines:
        parts = line.strip().split(" +++$+++ ")
        if len(parts) == 5:
            # Making sure that each part has proper structure
            lines_dict[parts[0]] = parts[4] 
        else:
            i += 1
            if i < 5:
                print(f"Skipping {parts[0]} as it's missing text: {parts}")
    pairs = []
    for con in conv:
        ids = eval(con.split(" +++$+++ ")[-1])
        # Extract ids of conversations (e.g. ['L194', 'L195', 'L196', 'L197'])
        for i in range(len(ids)):
            qa_pairs = []
            if i == len(ids) - 1:
                break
            try:
                first = lines_dict[ids[i]].strip()  
                second = lines_dict[ids[i+1]].strip() 
            except Exception as e:
                # print(f"Error processing pair for ids {ids[i]} and {ids[i+1]}: {e}")
                continue 
            qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
            qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
            pairs.append(qa_pairs)
    return pairs, lines_dict


def prepare_text_batches(pairs, batch_size=10000):
    os.makedirs('./datasets/MovieDialogsChunks', exist_ok=True)
    text_data = []
    file_count = 0
    for sample in tqdm.tqdm([x[0] for x in pairs]):
        text_data.append(sample)
        if len(text_data) == batch_size:
            with open(f'./datasets/MovieDialogsChunks/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1


def train_tokenizer(chunk_paths):
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )
    tokenizer.train(
        files=chunk_paths,
        vocab_size=30000,
        min_frequency=5,
        limit_alphabet=1000,
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[MASK]', '[UNK]', '[SEP]']
    )
    os.makedirs('./bert_tokenizer', exist_ok=True)
    tokenizer.save_model('./bert_tokenizer', 'bert-movie_corp')
    tokenizer = BertTokenizer.from_pretrained('./bert_tokenizer/bert-movie_corp-vocab.txt', local_files_only=True)