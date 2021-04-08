# coding=utf-8

import asyncio
import functools
from sanic import Sanic
from sanic.response import text
import json
import re
from sanic.log import logger
from sanic.exceptions import ServerError
import sanic

app = Sanic(__name__)

MAX_BATCH_SIZE = 32  # we put at most MAX_BATCH_SIZE things in a single batch
MAX_WAIT = 0.1  # we wait at most MAX_WAIT seconds before running for more inputs to arrive in batching

from pathlib import Path
import os, sys
import numpy as np
from utils import load_yaml, SENTI_ID_MAP_INV
from tokenizer import get_tokenizer
from model import *
import pickle
from dataset import TargetDependentExample
import traceback
import torch

device = "cpu"

base_dir = Path("./")
config_dir = base_dir / "config"


class ModelRunner:
    def __init__(self, version):
        self.queue = []
        self.queue_lock = None
        self.needs_processing = None
        self.needs_processing_timer = None
        self.version = version

        self.load_configs()
        self.load_model()
        self.load_tokenizer()
        self.model.eval()

    def load_configs(self):

        self.deploy_config = load_yaml(config_dir / f"deploy_{self.version}.yaml")
        self.model_dir = base_dir / "models" / self.deploy_config["model_dir"]
        self.model_config = load_yaml(self.model_dir / "model.yaml")
        self.train_config = load_yaml(self.model_dir / "train.yaml")
        self.state_path = self.model_dir / self.deploy_config["state_file"]
        self.model_class = self.train_config["model_class"]
        self.preprocess_config = self.model_config[self.model_class]["preprocess"]
        self.preprocess_config["text_preprocessing"] = ""
        self.body_config = self.model_config[self.model_class]["body"]

    def load_model(self):
        word2idx_info = pickle.load(open(self.model_dir / "word2idx_info.pkl", "rb"))
        self.word2idx = word2idx_info["word2idx"]
        self.emb_dim = word2idx_info["emb_dim"]
        emb_vectors = np.random.rand(len(self.word2idx), self.emb_dim)
        num_emb = len(self.word2idx)
        self.model = getattr(sys.modules[__name__], self.model_class)(
            model_config=self.body_config,
            num_labels=3,
            pretrained_emb=emb_vectors,
            num_emb=num_emb,
            pretrained_lm=None,
            device=device,
        )
        self.model.load_state(self.state_path)

    def load_tokenizer(self):
        self.tokenizer = get_tokenizer(
            source=self.preprocess_config["tokenizer_source"],
            name=self.preprocess_config["tokenizer_name"],
        )

    def schedule_processing_if_needed(self):
        if len(self.queue) >= MAX_BATCH_SIZE:
            logger.debug("next batch ready when processing a batch")
            self.needs_processing.set()
        elif self.queue:
            logger.debug("queue nonempty when processing a batch, setting next timer")
            self.needs_processing_timer = app.loop.call_at(
                self.queue[0]["time"] + MAX_WAIT, self.needs_processing.set
            )

    async def process_input(self, input):
        our_task = {
            "done_event": asyncio.Event(loop=app.loop),
            "input": input,
            "time": app.loop.time(),
        }
        async with self.queue_lock:
            self.queue.append(our_task)
            logger.debug("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed()

        await our_task["done_event"].wait()
        return our_task["output"]

    def run_model(self, input):  # runs in other thread
        with torch.no_grad():
            outputs = dict()
            results = self.model(
                **input,
            )
            outputs["sentiment_id"] = torch.argmax(results[1], dim=1)
            outputs["score"] = torch.nn.functional.softmax(results[1], dim=1)[:, outputs["sentiment_id"]]
            return outputs

    async def model_runner(self):
        self.queue_lock = asyncio.Lock(loop=app.loop)
        self.needs_processing = asyncio.Event(loop=app.loop)
        while True:

            await self.needs_processing.wait()
            self.needs_processing.clear()
            if self.needs_processing_timer is not None:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None

            async with self.queue_lock:
                if self.queue:
                    longest_wait = app.loop.time() - self.queue[0]["time"]
                    logger.debug(
                        "launching processing. queue size: {}. longest wait: {}".format(
                            len(self.queue), longest_wait
                        )
                    )
                else:  # oops
                    longest_wait = None
                    logger.debug(
                        "launching processing. queue size: {}. longest wait: {}".format(
                            len(self.queue), longest_wait
                        )
                    )
                    continue

                to_process = self.queue[:MAX_BATCH_SIZE]
                del self.queue[: len(to_process)]
                self.schedule_processing_if_needed()
            # so here we copy, it would be neater to avoid this

            input = dict()
            for col in self.model.INPUT_COLS:

                if col in to_process[0]["input"].features:
                    if "label" not in col:
                        input[col] = torch.stack(
                            [t["input"].features[col] for t in to_process], dim=0
                        ).to(device)

            results = await app.loop.run_in_executor(
                None, functools.partial(self.run_model, input)
            )

            results["sentiment"] = [
                SENTI_ID_MAP_INV[p]
                for p in results["sentiment_id"].detach().cpu().numpy()
            ]
            results["score"] = [float(s[0]) for s in results["score"].detach().cpu().numpy()]

            for i in range(len(to_process)):
                output = dict()
                t = to_process[i]
                output["sentiment_id"] = int(results["sentiment_id"][i])
                output["sentiment"] = results["sentiment"][i]
                output["score"] = results["score"][i]
                t["output"] = output
                t["done_event"].set()
            del to_process


def parse_input(data):
    results = []
    for x in data['doclist'][0]['labelunits']:
        row = dict()
        st_idx = x['unit_index'][0]
        row['unit_text'] = x['unit_text']
        row['subject_index'] = [[i[0] - st_idx, i[1] - st_idx] for i in x['subject_index']]
        row['aspect_index'] = [[i[0] - st_idx, i[1] - st_idx] for i in x['aspect_index']]

        results.append(row)
    return results

def insert_sentiment(data, sentiments):
    for i, x in enumerate(data['doclist'][0]['labelunits']):
        x['sentiment'] = sentiments[i]

    return data


def is_spam(unit_text, subject_text):
    clean_text = ' '.join(re.sub("(#\s?\w+)|(@\w+)|(•\s?\w+)"," ",unit_text).split())
    for t in subject_text:
        if t in clean_text:
            return False
    return True


model_runner = ModelRunner("chinese")

@app.route("/rolex_sentiment", methods=["POST"], stream=True)
async def target_sentiment(request):
    try:
        raw_data = await request.stream.read()
        raw_data = json.loads(raw_data)

        parsed_data = parse_input(raw_data)

        response = dict()

        if raw_data["language"] == "english":
            response["message"] = "OK"
            response["sentiment"] = "not_supported"

        elif raw_data["language"] == "chinese":

            sentiments = []
            for x in parsed_data:

                # rule check
                if not is_spam(x['unit_text'], [x['unit_text'][i[0]:i[1]] for i in x["subject_index"]]):

                    # model
                    example = TargetDependentExample(
                        raw_text=x["unit_text"],
                        target_locs=x["subject_index"] + x["aspect_index"],
                        tokenizer=model_runner.tokenizer,
                        preprocess_config=model_runner.preprocess_config,
                        required_features=model_runner.model.INPUT_COLS,
                        word2idx=model_runner.word2idx,
                    )

                    results = await model_runner.process_input(example)
                    sentiments.append(results["sentiment"])
                else:
                    sentiments.append("neutral")

            raw_data = insert_sentiment(raw_data, sentiments)

        return sanic.response.json(raw_data, status=200)

    except Exception as e:
        msg = traceback.format_exc()
        return sanic.response.json({"message": msg}, status=500)

app.add_task(model_runner.model_runner())
app.run(host="0.0.0.0", port=8080, debug=False)
