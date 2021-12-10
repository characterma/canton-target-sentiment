import os
import json
import streamlit as st
from pathlib import Path
os.chdir("../src")
from pipeline import Pipeline


st.set_page_config(
    page_title="NLP Pipeline", layout="wide"
)


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
            content = st.text_input(
                "content", ""
            )

            explanation = st.checkbox(
                "Explanation", 
                value=False, 
            )

            if st.button("Execute"):

                prediction = pipeline.predict({
                    "content": content
                })

                prediction

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