from data.dataset import BERTDataset
from data.preprocessing import *
from model.bert import BERTModel
from training.train import train
from training.optimizer import create_optimizer
from utils.helpers import save_model, load_model

from data.get_data import download_corpus

def download_files():
    print("Downloading began")
    url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
    local_fname = url.split('/')[-1]  
    download_corpus(url, local_fname)

    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    local_fname = url.split('/')[-1]  
    download_corpus(url, local_fname)


def preprocess_data():
    corpus_movie_conv = 'datasets/cornell movie-dialogs corpus/movie_conversations.txt'
    corpus_movie_lines = 'datasets/cornell movie-dialogs corpus/movie_lines.txt'
    with open(corpus_movie_conv, 'r', encoding='iso-8859-1') as c:
        conv = c.readlines()
    with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
        lines = l.readlines()
    pairs, lines_dict = prepare_qa_pairs(lines, conv)
    prepare_text_batches(pairs)
    chunk_paths = [os.path.join('./datasets/MovieDialogsChunks/', x) for x in os.listdir('./datasets/MovieDialogsChunks')]
    train_tokenizer(chunk_paths)


if __name__ == "__main__":
    # download_files()
    preprocess_data()
