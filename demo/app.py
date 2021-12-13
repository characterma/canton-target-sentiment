import os
import json
import requests
import streamlit as st
from streamlit import components
from pathlib import Path
if os.getcwd().endswith("demo"):
    os.chdir("../src")
from pipeline import Pipeline
from explain.utils import visualize_data_record_bert, get_explanation, get_segment_level_explanation


st.set_page_config(
    page_title="NLP Pipeline", layout="wide"
)


def get_word_segments(raw_text):

    data = requests.post("https://ai.wws.wisers.net/ltp-commercial/ltp/article/single/pos", json={'content': raw_text}).json()
    segments = []
    segments_idxs = []
    for sent_res in data['content']['sent_res']:
        segments += sent_res['words']
        segments_idxs += sent_res['art_indexs']
    return segments, segments_idxs


def detect_valid_model_dirs():
    model_file_names = ["label_to_id.json", "model.pt", "model.yaml", "run.yaml"]
    model_dir_names = ["tokenizer"]
    valid_model_dirs = []
    for dir_path, dir_names, file_names in os.walk('../output'):
        dir_path = Path(dir_path)
        if set(model_file_names).issubset(set(file_names)) and set(model_dir_names).issubset(set(dir_names)):
            valid_model_dirs.append(dir_path.parent)
    return valid_model_dirs


def train_model_page():
    st.write('Train a new model.') 


def load_model_page():
    st.write("Load an existing model.") 

    valid_model_dirs = [None] + detect_valid_model_dirs()
    model_dir = st.selectbox(
        'Pick a model directory.',
        tuple(valid_model_dirs), 
        index=0
    )

    if model_dir is not None:
        pipeline = Pipeline(model_dir=model_dir) 
    else:
        pipeline = None

    if pipeline is not None:
        with st.expander("Predict"):
            content = st.text_input("content", "")

            raw_data = {'content': content}

            if st.button("Predict"):

                prediction = pipeline.predict(raw_data)
                st.write(prediction)

            if st.button("Interpret"):
                tokens, scores, attr_target, attr_target_prob, tokens_encoded, faithfulness = get_explanation(
                    pipeline, 
                    raw_data, 
                    enable_faithfulness=True
                )
                
                st.write(f"Faithfulness (COMP) = {faithfulness}")
                tokens_seg, scores_seg = get_segment_level_explanation(
                    seg_func=get_word_segments, 
                    raw_text=raw_data['content'], 
                    tokens=tokens, 
                    scores=scores, 
                    tokens_encoded=tokens_encoded
                )

                vis = visualize_data_record_bert(
                    pipeline, 
                    raw_data, 
                    tokens=tokens_seg, 
                    scores=scores_seg, 
                    attr_target=attr_target, 
                    attr_target_prob=attr_target_prob
                )

                # vis = visualize_data_record_bert(
                #     pipeline, 
                #     raw_data, 
                #     tokens=tokens, 
                #     scores=scores, 
                #     attr_target=attr_target, 
                #     attr_target_prob=attr_target_prob
                # )

                components.v1.html(
                    vis._repr_html_(), scrolling=True, height=250
                )


        with st.expander("Evaluate"):
            uploaded_file = st.file_uploader("Choose a json file")
            if uploaded_file is not None:
                # Can be used wherever a "file-like" object is accepted:
                json.load(uploaded_file)

                print(len(json))


if __name__ == "__main__":
    tabs = [
        "Train a new model",
        "Load an existing model"
    ]
    page = st.sidebar.radio("Tabs",tabs)

    if page == "Train a new model":
        train_model_page()
    elif page == "Load an existing model":
        load_model_page()