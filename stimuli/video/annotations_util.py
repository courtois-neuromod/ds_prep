import glob
import json
import random
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from bpemb import BPEmb
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("wordnet")


def load_json_file(datapath: str):
    """Loads json files produced by AssemblyAI."""
    with open(datapath, "r", encoding="utf-8") as file:
        data = json.load(file)
    transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcript


def concat_transcript(json_dir: str):
    """Concats the transcripts from json files."""
    json_files = sorted(json_dir.glob("*.json"))
    transcript = []
    transcript_text = ""
    for json_file in json_files:
        segment = load_json_file(json_file)
        transcript_text += segment
    transcript = transcript_text.strip()
    return transcript


# found outlawed words
def find_outlaw(word):
    """Find words that contain a same character 3+ times in a row."""
    is_outlaw = False
    for i, letter in enumerate(word):
        if i > 1:
            if word[i] == word[i - 1] == word[i - 2] and word[i].isalpha():
                is_outlaw = True
                break
    return is_outlaw


def word_characteristics(transcript):
    # concatanate all strings from all seasons
    words = re.findall("\w+", transcript)
    unique_words = set(re.findall(r"\w+", transcript, re.UNICODE))
    return words, unique_words


def word_statistics(transcript):

    freq_splits = {}
    unique_res = {}
    outlaws = {}
    outlaws = [s for s in transcript if find_outlaw(s)]
    res = np.array(outlaws)
    unique_res = np.unique(res)
    words = re.findall("\w+", transcript)
    freq_splits = FreqDist(words)
    frequent_word_list = []
    frequency_word_list = []

    for i in range(20):
        frequent_word_list.append(freq_splits.most_common(20)[i][0])
        frequency_word_list.append(freq_splits.most_common(20)[i][1])

    return (
        freq_splits,
        unique_res,
        frequent_word_list,
        frequency_word_list,
        outlaws,
    )


def stopword_statistics(tokens_norm):
    stop_words = set(stopwords.words("english"))
    for season in range(1, 11):
        freq_stopwords = [(sw, tokens_norm[season].count(sw)) for sw in stop_words]
        freq_stopwords.sort(key=lambda x: x[1], reverse=True)
    return stop_words, freq_stopwords


def token_statistics(transcript):
    tokens_norm = {}
    unique_tokens = {}
    unique_tokens_list = {}
    total_number_of_tokens = 0
    bpemb_en = BPEmb(lang="en")
    lemmatiser = WordNetLemmatizer()
    tokens = bpemb_en.encode(transcript)
    tokens_norm = [
        lemmatiser.lemmatize(t.lower(), "v") for t in bpemb_en.encode(transcript)
    ]

    # Use set to find unique tokens
    unique_tokens = set(tokens)
    # Convert the set back to a list if needed
    unique_tokens_list = list(unique_tokens)
    total_number_of_tokens += len(tokens)
    return (
        tokens,
        unique_tokens_list,
        tokens_norm,
        total_number_of_tokens,
    )


def ngram_extractor(transcript, n_gram, tokens_norm):
    stop_words, _ = stopword_statistics(tokens_norm)
    token = [
        token
        for token in transcript.lower().split(" ")
        if token != ""
        if token not in stop_words
    ]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(transcript, n_gram, max_row, tokens_norm):
    temp_dict = defaultdict(int)

    for word in ngram_extractor(transcript, n_gram, tokens_norm):
        temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(
        max_row
    )
    temp_df.columns = ["word", "wordcount"]
    return temp_df


def plot_corpus_characteristics(
    value, title: str, xlabel: str, ylabel: str, tilt: False, type: str
):
    plt.figure(figsize=(6, 4))

    if type == "bar":
        # plt.bar(data_range, value, color="#4958B5")
        sns.barplot(x=list(value.keys())[0], y=list(value.keys())[1], data=value)
    elif type == "count":
        sns.countplot(y=value)
    elif type == "ngram":
        fig, ax = plt.subplots()
        y_pos = np.arange(len(value))
        ax.barh(y_pos, value[list(value.keys())[1]])
        ax.set_yticks(y_pos, labels=value[list(value.keys())[0]])
        ax.invert_yaxis()

    plt.title(title, fontsize=12)
    plt.ylabel(ylabel, fontsize=10)
    plt.xlabel(xlabel, fontsize=10)
    if tilt == True:
        plt.xticks(rotation=45)

    plt.show()
