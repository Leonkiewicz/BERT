import urllib.request
import tarfile
import zipfile
import os
import pandas as pd


def download_corpus(url, local_fname):
    extract_path = 'datasets'
    os.makedirs(extract_path, exist_ok=True)
    urllib.request.urlretrieve(url, local_fname)
    print(f"File downloaded as {local_fname}")
    if local_fname.endswith('.tar.gz') or local_fname.endswith('.tgz'):
        with tarfile.open(local_fname, 'r:gz') as tar:
            tar.extractall(os.path.join(os.getcwd(), extract_path))
            print("Extracted tar.gz file.")
        os.remove(local_fname)
        print(f"Deleted {local_fname}, only extracted files are kept.")
    elif local_fname.endswith('.zip'):
        with zipfile.ZipFile(local_fname, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(os.getcwd(), extract_path))
            print("Extracted zip file.")
        os.remove(local_fname)
        print(f"Deleted {local_fname}, only extracted files are kept.")
    else:
        print("Unsupported file format.")


def text_to_df(file_path):
    labels = {'pos': 1, 'neg': 0}
    data = []
    for sub_dir in ['neg', 'pos']:
        dir_path = os.path.join(file_path, sub_dir)
        label = labels[sub_dir]
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    data.append({'review': content, 'label': label})
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    print("Downloading began")
    # url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
    # local_fname = url.split('/')[-1]  
    # download_corpus(url, local_fname)

    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    local_fname = url.split('/')[-1]  
    download_corpus(url, local_fname)