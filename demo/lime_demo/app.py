import requests
import numpy
import streamlit as st
import streamlit.components.v1 as components
from multiprocessing import Pool
from functools import partial
from lime.lime_text import LimeTextExplainer

class_names = ['positive', 'neutral', 'negative']



def get_sentiment_prediction(raw_text, endpoint='http://aiapi.wisers.com/sentiment-dl-common-gpu/sentiment/document/zh'):
    try:
        result = requests.post(endpoint, json={'text': raw_text}).json()
        output = [result['data']['scores'][c]  for c in class_names]
    except:
        output=[0,1,0]
        #print ("raw_text", raw_text)
        #print ('result', result)
    return output


def filter_sentences_by_label(raw_text_list, endpoint='http://aiapi.wisers.com/sentiment-dl-common-gpu/sentiment/document/zh', label='negative'):
    sents = []
    for raw_text in raw_text_list: 
        result = requests.post(endpoint, json={'text': raw_text}).json()
        if result['data']['label']==label:
            sents.append(raw_text)
    return sents


def filter_sentences_by_keyword(text, keywords=['苏宁','家乐福']):
    data = requests.post("https://ai.wws.wisers.net/ltp-commercial/ltp/article/single/pos", json={'content': text}).json()
    #print ("data", data)
    sentences = data['content']['sentences']
    result = []
    for sent in sentences:
        for k in keywords:
            if k in sent:
                result.append(sent)
                break
    return list(set(result))


def get_sentiment_prediction_wrapper_process(raw_text_list, endpoint='http://aiapi.wisers.com/sentiment-dl-common-gpu/sentiment/document/zh'):
    get_result = partial(get_sentiment_prediction, endpoint=endpoint)
    output = []
    with Pool(20) as p:
    # schedule one map/worker for each row in the original data
        q = p.map(get_result, raw_text_list)
    return numpy.array(q)


# Define page settings
st.title('LIME explainer app for classification models')

input_text = st.text_area('Enter your text:', "")
endpoint = st.text_input('Sentiment API endpoint', value='http://aiapi.wisers.com/sentiment-dl-common-gpu/sentiment/document/zh')
filter_keyword = st.text_input('Filter keywords list', value="苏宁,家乐福")
filter_keyword = filter_keyword.split(',')
filter_label = st.text_input('Filter sentence by sentiment labels (negative/positive/neutral):', value="negative")
n_samples = st.number_input('Number of samples to generate for LIME explainer: (For really long input text, go up to 5000)', value=500)

#print ('input_text', input_text)
sentences_kw = filter_sentences_by_keyword(input_text, keywords=filter_keyword)
sentences_kw_label = filter_sentences_by_label(sentences_kw, endpoint=endpoint, label=filter_label)
st.write(sentences_kw_label)

explain_text = st.text_area('Enter your text for explaination:', "")

if st.button("Explain Results"):
    with st.spinner('Calculating...'):
        explainer = LimeTextExplainer(class_names=class_names, char_level=True,  random_state=32)
        exp = explainer.explain_instance(explain_text, get_sentiment_prediction_wrapper_process, num_features=10, num_samples=n_samples, top_labels=1)

        # Display explainer HTML object
        components.html(exp.as_html(), height=800)

        

