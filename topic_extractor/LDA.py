import json
import re

from gensim.models import CoherenceModel, LdaModel
from matplotlib import pyplot
from pyLDAvis import PreparedData
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import nltk
import gensim.corpora as corpora
from pprint import pprint
import numpy as np

import pyLDAvis.gensim
import pickle
import pyLDAvis
import os
import time
import webbrowser
from random import randint

nltk.download('stopwords')
from nltk.corpus import stopwords


# get data from the transcription source
# Loading data
def load_data(path):
    # review_data = pd.read_csv("spt2.txt", sep=" ")
    print("ppath: " + path)
    f = open(path, "r+")
    transcribed_data = f.read()
    # print(f'transcribed_data::: \n {transcribed_data}')

    print(f'length of transcribed_data {len(transcribed_data)}')
    return transcribed_data


'''first analyze using a wordcloud'''


def analyze_text(data):
    data = data.split(' ')
    # Data cleaning
    data = list(map(lambda x: re.sub('[,\.!?]', '', x), data))
    data = list(map(lambda x: x.lower(), data))
    # print(f'preprocessed::: \n {transcribed_data}')

    transcribed_data_as_string = ''
    for word in data:
        transcribed_data_as_string += word + ','

    wordcloud = WordCloud(background_color="white", max_words=10000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(transcribed_data_as_string)
    wordcloud.scale = 3
    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# Exploratory analysis
# analyze_text(transcribed_data)

def make_sentences_form_transcribed_data(transcribed_data):
    transcribed_data_as_sentences = nltk.sent_tokenize(transcribed_data)

    # transcribed_data_as_sentences = re.findall('.*?([.?\s!]\s)|^(\b\w)', transcribed_data)
    # transcribed_data_as_sentences = list(
    #     filter(lambda x: x != '. ' and x != '? ' and x != '! ' and x != None, transcribed_data_as_sentences))
    # transcribed_data_as_sentences = [string for string in transcribed_data_as_sentences if string != ""]
    # print(transcribed_data_as_sentences)
    return transcribed_data_as_sentences


# Preparing data for LDA analysis
def prepare_data_for_lda_analysis(transcribed_data):
    stop_words = stopwords.words('english')

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    # make sentences form transcribed data
    transcribed_data_as_sentences = make_sentences_form_transcribed_data(transcribed_data)

    for sentence in transcribed_data_as_sentences:
        print(f'sentence: {sentence}')

    data_words = list(sent_to_words(transcribed_data_as_sentences))

    # remove stop words
    data_words = remove_stopwords(data_words)
    return data_words


# LDA model training ##################
def train_lda_model(corpus, lda_model):
    print("corpus:" + str(len(corpus)))
    # View
    # print(corpus)
    # print(corpus[:1][0][:100])

    print(f'lda_model.num_topics = {lda_model.num_topics}')

    return lda_model, lda_model[corpus]


# Analyzing LDA model results ##################
def analyze_lda_model_results(lda_model, corpus, id2word, num_topics):
    rand = str(randint(0, 10000))
    # Visualize the topics
    # pyLDAvis.enable_notebook()
    current_dir = os.path.abspath(os.curdir)
    LDAvis_data_filepath = current_dir + '/AppData/tempStorage/voiceEnhancerTranscriptions/html/ldavis_prepared_' + rand + '_' + str(
        num_topics)
    LDAvis_json_filepath = current_dir + '/AppData/tempStorage/voiceEnhancerTranscriptions/json/ldavis_prepared_' + rand + '_' + str(
        num_topics) + ".json"
    print(f'LDAvis_data_filepath: {LDAvis_data_filepath}')
    print(f'os.path: {os.path}')
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    print(f' str(num_topics) "{str(num_topics)}')
    # if np.iscomplexobj(LDAvis_prepared):
    #     LDAvis_prepared = abs(LDAvis_prepared)
    # pyLDAvis.save_html(LDAvis_prepared, current_dir + '/AppData/tempStorage/voiceEnhancerTranscriptions/html/ldavis_prepared_'  + rand+'_' + str(num_topics) + '.html')
    print('Done!. analyze_lda_model_results saved in ./results ')
    # pyLDAvis.save_json(LDAvis_prepared, LDAvis_json_filepath)
    print(type(LDAvis_prepared))
    pyLDAvis.show(LDAvis_prepared, open_browser=True)
    # webbrowser.open(current_dir + '/AppData/tempStorage/voiceEnhancerTranscriptions/html/ldavis_prepared_' +rand+'_' + str(num_topics) + '.html', new=1)


def run_lda_topic_extraction(transcribed_text, analyze=False):
    preprocessed_data_words = prepare_data_for_lda_analysis(transcribed_text)
    print(preprocessed_data_words)

    # Create Dictionary
    id2word = corpora.Dictionary(preprocessed_data_words)
    lda_model = ''
    # Create Corpus
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in preprocessed_data_words]

    num_topics_of_max_coherence = None
    # num_topics_of_max_coherence = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=preprocessed_data_words, start=2, limit=40, step=6)

    # number of topics (set default to 30)
    num_topics = num_topics_of_max_coherence if num_topics_of_max_coherence is not None else 30

    # Build LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)

    lda_model, doc_lda = train_lda_model(corpus, lda_model)
    words_with_topic_biased = {}
    repeating_words_with_count = {}

    # print(f'top_topics:{lda_model.top_topics(corpus=corpus)}')
    # print(f'show_topics:{lda_model.show_topics()}')
    # print(f'get_topics:{lda_model.get_topics()}')
    # print(f'log_perplexity:{lda_model.log_perplexity(chunk=corpus)}')

    def calculate_topic_biases():
        topics = lda_model.print_topics()
        print(f'topics: {topics}')
        for topic in topics:
            print(type(topic))
            topic_biases = topic[1].split(" + ")
            for topic_bias_obj in topic_biases:
                topic_bias_obj_as_arr = topic_bias_obj.split('*"')
                word = topic_bias_obj_as_arr[1].rstrip('"')
                value = topic_bias_obj_as_arr[0]

                if repeating_words_with_count.get(word) is None:
                    repeating_words_with_count[word] = [float(value)]
                    print(f'adding {word}: {float(value)}')

                else:
                    current_occurences = repeating_words_with_count.get(word)
                    current_occurences.append(float(value))
                    repeating_words_with_count[word] = current_occurences
                    print(f'adding  to current `{word}`: {float(value)}')

                    # repeating_words_with_count[word] = repeating_words_with_count[word] + 1
                    # print("'" + word + "' incremented at " + str(repeating_words_with_count[word]))
                    #
                    # print(f'Before averaging: {word} : {words_with_topic_biased[word]}')
                    # print(f"Executing avg =  {float(value)} + {float(words_with_topic_biased[word]) } / {repeating_words_with_count[word]}")
                    # words_with_topic_biased[word] = \
                    #     round(float(value) + float(words_with_topic_biased[word]) / repeating_words_with_count[word], 4)
                    # print(f'After averaging: {word} : {words_with_topic_biased[word]}')

        print(f'repeating_words_with_count : {repeating_words_with_count}')
        for repeating_word in repeating_words_with_count:
            value_array = repeating_words_with_count.get(repeating_word)
            if len(value_array) > 1:
                words_with_topic_biased[repeating_word] = round(float((sum(value_array) / len(value_array))), 4)
                print(
                    f"Executing: avg of `{repeating_word}` =  sum of {value_array} / {len(value_array)} = {words_with_topic_biased[repeating_word]}")
            else:
                print(f"`{repeating_word}` occurred only once")
                words_with_topic_biased[repeating_word] = value_array[0]

        print("Sorting by biased value")
        print("\n \n")
        words_with_topic_biased_sorted = dict(
            sorted(words_with_topic_biased.items(), key=lambda item: item[1], reverse=True))
        print(words_with_topic_biased_sorted)
        return words_with_topic_biased_sorted

    def extract_most_salient_terms():
        # extracting Most Salient Terms (Default)
        # https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

        rt = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, R=30)
        import pandas as pd
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(rt[1])
        dr = rt[1][rt[1].Category == 'Default'] \
            .drop(['Total', 'Category', 'logprob', 'loglift'], axis=1)
        drr = dr.to_dict('index')
        print(f"default: {drr}")
        dictt = {}
        for key in drr:
            item = drr.get(key)
            dictt[str(item.get("Term"))] = item.get("Freq")
        print(f"dict: {dictt}")

        return dictt

    if analyze:
        pass
        # analyze_lda_model_results(lda_model, corpus, id2word, num_topics)

    return calculate_topic_biases()
    # return extract_most_salient_terms()

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    x = range(start, limit, step)
    num_topics_of_max_coherence = x[coherence_values.index(max(coherence_values))]
    print('num_topics_of_max_coherence: ', num_topics_of_max_coherence)

    # import matplotlib.pyplot as plt
    # plt.plot(x, coherence_values)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc='best')
    # plt.show()

    # return model_list, coherence_values
    return num_topics_of_max_coherence
