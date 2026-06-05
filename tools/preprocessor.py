# import os
# os.environ['GENSIM_DATA_DIR'] = '/home/isel-har/goinfre/gensim'
import nltk
from transformers import AutoTokenizer
import contractions
import string
from nltk.stem     import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from nltk.corpus   import wordnet
import numpy as np
import gensim.downloader as api

import pkg_resources
from symspellpy import SymSpell, Verbosity



nltk.download("averaged_perceptron_tagger_eng")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")


class NLProcessor:
    punct_translator = str.maketrans("", "", string.punctuation)
    digit_translator = str.maketrans("", "", string.digits)

    def __init__(
        self,
        use_stopwords=False,
        normalize=True,
        lower=True,
        use_clean=True,
        remove_punc=True,
        use_spell=True,
        embedder=api.load('word2vec-google-news-300')
    ):
        self.use_clean     = use_clean
        self.use_stopwords = use_stopwords
        self.use_normalize = normalize
        self.padding_len   = 0
        self.unfound_words = []
        self.remove_punc   = remove_punc
        self.lower         = lower
 

        self.stop_words = set(stopwords.words("english")) if use_stopwords else None
        self.normalizer = WordNetLemmatizer() if normalize else None
        self.embedder = embedder
        self.sym_spell =  SymSpell(max_dictionary_edit_distance=2, prefix_length=7) if use_spell else None
        if self.sym_spell:
            dictionary_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


    @staticmethod
    def get_wordnet_pos(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN


    def clean(self, sentences):
        cleaned_tweets = []

        for sentence in sentences:
  

            sentence = contractions.fix(sentence)
            if self.lower:
                sentence = sentence.lower()
            if self.remove_punc:
                sentence = sentence.translate(self.punct_translator)
            sentence = sentence.translate(self.digit_translator)
            sentence = sentence.strip() # remove leading/trailing spaces
            sentence = " ".join(sentence.split()) # remove duplicate spaces

            if len(sentence) == 0:
              print("empty string!")
            
            cleaned_tweets.append(sentence)

        return cleaned_tweets
    

    def tokenization(self, sentences):
        tokens_list = list()

        for text in sentences:
            tokens = word_tokenize(text)

            if len(tokens) == 0:
              print("empty tokens")
            tokens_list.append(tokens)

        return tokens_list


    def filter_stopwords(self, tokens_list):
        filtered_tokens = []

        for tokens in tokens_list:
            filtered = [word for word in tokens if word not in self.stop_words]

            if len(filtered) == 0:
              filtered_tokens.append(tokens)
            else:
              filtered_tokens.append(filtered)

        return filtered_tokens


    def spell_correct(self, tokens_list):
        corrected_tokens = []

        for tokens in tokens_list:
          corrected_text = []

          for word in tokens:
            if len(word) <= 2:
                corrected_text.append(word)
                continue

            suggestions = self.sym_spell.lookup(
                word, Verbosity.CLOSEST, max_edit_distance=2
            )
  
            corrected_word = suggestions[0].term if suggestions else word

            corrected_text.append(corrected_word)


          if len(corrected_text) == 0:
            print("empty corrected text!")
          corrected_tokens.append(corrected_text)

        return corrected_tokens




    def lemmatize(self, tokens_list):

        lemmatized_tokens_list = list()

        for tokens in tokens_list:
            lemmas = list()
            pos_tags = nltk.pos_tag(tokens)
            for word, tag in pos_tags:
                wn_tag = self.get_wordnet_pos(tag)
                lemma = self.normalizer.lemmatize(word, wn_tag)
                lemmas.append(lemma)
            
            if len(lemmas) == 0:
              print("empty lemmas tokens")
            lemmatized_tokens_list.append(lemmas)

        return lemmatized_tokens_list


    def add_padding(self, embedded_tokens):
        
        padding_vec = np.zeros(embedded_tokens[0][0].shape[0])
        padded_list = list()
        for embedded_token in  embedded_tokens:
  
            for _ in range(len(embedded_token), self.padding_len):
                embedded_token.append(padding_vec)

            padded_list.append(np.array(embedded_token))

        return np.array(padded_list)


    def word2vec_embedding(self, tokens_list):
        embedded_tokens = list()

        for i, tokens in enumerate(iterable=tokens_list):
            
            embeddings = list()
            for token in tokens:

                if token in self.embedder:
                    embedding_vec = self.embedder[token]
                    embeddings.append(embedding_vec)
                else:
                    print(f"token not found : {token}")

            if len(embeddings) > self.padding_len:
                self.padding_len = len(embeddings)


            if len(embeddings) > 0:
                embedded_tokens.append(embeddings)
            else:
                self.unfound_words.append(i)
        return embedded_tokens


    def transform(self, raw_sentences):

        sentences = raw_sentences
        if self.use_clean:
            sentences = self.clean(sentences)

        tokens  = self.tokenization(sentences)
        if self.use_stopwords:
            tokens  = self.filter_stopwords(tokens)

        if self.sym_spell:
            tokens = self.spell_correct(tokens)

        if self.use_normalize:
            tokens  = self.lemmatize(tokens)


        embedded_tokens = self.word2vec_embedding(tokens)
        return self.add_padding(embedded_tokens)
