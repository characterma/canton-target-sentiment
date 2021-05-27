# coding=utf-8
import argparse
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

from pathlib import Path
import os, sys
import numpy as np
import pickle
from dataset import TargetDependentExample
from neg_kws import neg_kws
import traceback
import torch
from run import load_config, init_model, init_tokenizer
from trainer import evaluation_step


MAX_WAIT = 0.1


class ModelRunner:
    """
    """
    def __init__(self, args):
        self.args = load_config(args=args)
        self.model_config = self.args.model_config
        self.prepro_config = self.args.prepro_config

        self.tokenizer = init_tokenizer(args=args)
        self.model = init_model(args=args)
        self.batch_size = self.args.eval_config['batch_size']

        self.queue = []
        self.queue_lock = None
        self.needs_processing = None
        self.needs_processing_timer = None

    def schedule_processing_if_needed(self):
        if len(self.queue) >= self.batch_size:
            self.needs_processing.set()
        elif self.queue:
            self.needs_processing_timer = app.loop.call_at(
                self.queue[0]["time"] + MAX_WAIT, self.needs_processing.set
            )

    async def process_input(self, input):
        task = {
            "done_event": asyncio.Event(loop=app.loop),
            "time": app.loop.time(),
            "input": input,
        }
        async with self.queue_lock:
            self.queue.append(task)
            logger.debug("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed()

        await task["done_event"].wait()
        return task["output"]

    def make_batch(self, to_process):
        batch = dict()
        for col in to_process[0]["input"].feature_dict:
            batch[col] = torch.stack([t["input"].feature_dict[col] for t in to_process], dim=0)
        return batch

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
                else: 
                    longest_wait = None
                    continue
                to_process = self.queue[:self.batch_size]
                del self.queue[: len(to_process)]
                self.schedule_processing_if_needed()

            batch = self.make_batch(to_process)
            batch_results = await app.loop.run_in_executor(
                None, functools.partial(evaluation_step, model=self.model, batch=batch, device=self.args.device)
            )

            for i in range(len(to_process)):
                t = to_process[i]
                t["output"] = {"sentiment": batch_results["sentiment"][i], "score": batch_results["score"][i]}
                t["done_event"].set()
            del to_process


def parse_doc(doc, io_format):
    if io_format=='syntactic':
        results = []
        for x in doc['labelunits']:
            row = dict()
            st_idx = x['unit_index'][0]
            row['unit_text'] = x['unit_text']
            row['subject_index'] = [[i[0] - st_idx, i[1] - st_idx] for i in x['subject_index']]
            row['aspect_index'] = [[i[0] - st_idx, i[1] - st_idx] for i in x['aspect_index']]

            results.append(row)
        return results
    else:
        return [doc]


def parse_labelunit(u, io_format):
    if io_format=='syntactic':
        text = u['unit_text']
        target_locs = u["subject_index"] + u["aspect_index"]
        subj_text = [u['unit_text'][i[0]:i[1]] for i in u["subject_index"]]
    else:
        text = u['text']
        target_locs = u["target"]
        subj_text = [u['unit_text'][i[0]:i[1]] for i in target_locs]
    return text, target_locs, subj_text


def insert_sentiment(doc, sentiments, debugs=None):
    for i, x in enumerate(doc['labelunits']):
        x['sentiment'] = sentiment_to_id[sentiments[i]]

        if debugs is not None:
            x['debug'] = debugs[i]


def is_spam(unit_text, subject_text):
    clean_text = ' '.join(re.sub("(#\s?\w+)|(@\w+)|(•\s?\w+)"," ",unit_text).split())
    for t in subject_text:
        if t in clean_text:
            return False
    return True


@app.route("/target_sentiment", methods=["POST"], stream=True)
async def target_sentiment(request):
    try:
        raw_data = await request.stream.read()
        raw_data = json.loads(raw_data)
        language = raw_data["language"]
        io_format = raw_data["format"]
        documents = raw_data['doclist'] if io_format=='syntactic' else [raw_data]

        for doc in documents:

            parsed_doc = parse_doc(doc, io_format=io_format)
            sentiments = []
            debugs = []
            for u in parsed_doc:

                text, target_locs, subj_text = parse_labelunit(u, io_format=io_format)

                # 1. check spam
                # 2. model
                # 3. check negative kws
                debug = {}
                if not is_spam(text, subj_text):

                    data_dict = {
                        'content': text, 
                        'target_locs': target_locs, 
                    }
                    x = TargetDependentExample(
                        data_dict=data_dict,
                        tokenizer=model_runner.tokenizer,
                        prepro_config=model_runner.prepro_config,
                        required_features=model_runner.model.INPUT,
                        max_length=model_runner.model_config["max_length"],
                        diagnosis=True
                    )
                    results = await model_runner.process_input(x)

                    if results["sentiment"]=='negative':
                        # check neg kws:
                        if re.search(neg_kws, text) is None:
                            results["sentiment"] = "neutral"
                            debug['no_neg_kws'] = True

                    sentiments.append(results["sentiment"])
                else:
                    debug['is_spam'] = True

                    
                    sentiments.append("neutral")

                debugs.append(debug)

            insert_sentiment(doc, sentiments, debugs=debugs)

        return sanic.response.json(raw_data, status=200)

    except Exception as e:
        msg = traceback.format_exc()
        return sanic.response.json({"message": msg}, status=500)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../config/")
    args = parser.parse_args()

    sentiment_to_id = {
        "neutral": "0",
        "negative": "-1", 
        "positive": "1"
    }

    model_runner = ModelRunner(
        args=args
    )
    app.add_task(model_runner.model_runner())
    app.run(host="0.0.0.0", port=8080, debug=False)
