# coding=utf-8

import asyncio
import itertools
import functools
from sanic import Sanic
from sanic.response import text
import json
from sanic.log import logger
from sanic.exceptions import ServerError

import sanic
import threading

app = Sanic(__name__)

MAX_BATCH_SIZE = 32  # we put at most MAX_BATCH_SIZE things in a single batch
MAX_WAIT = 0.1  # we wait at most MAX_WAIT seconds before running for more inputs to arrive in batching

import logging
from pathlib import Path
import os, sys
import pandas as pd
import numpy as np
from transformers_utils import PretrainedLM
from utils import load_yaml, parse_api_req, SENTI_ID_MAP_INV
from tokenizer import get_tokenizer
from model import *
from dataset import TargetDependentExample
from explanation import LimeExplanation, AttnExplanation
import traceback
import torch

# device = 0
device = 1


def init_model(
    model_class,
    body_config,
    num_labels,
    pretrained_emb,
    num_emb,
    pretrained_lm,
    device,
    state_path=None,
):

    MODEL = getattr(sys.modules[__name__], model_class)
    model = MODEL(
        model_config=body_config,
        num_labels=num_labels,
        pretrained_emb=pretrained_emb,
        num_emb=num_emb,
        pretrained_lm=pretrained_lm,
        device=device,
    )
    if state_path is not None:
        model.load_state(state_path)
    return model


base_dir = Path("../")
config_dir = base_dir / "config"
deploy_config = load_yaml(config_dir / "deploy.yaml")

model_dir = base_dir / "models" / deploy_config["model_dir"]

model_config = load_yaml(model_dir / "model.yaml")
train_config = load_yaml(model_dir / "train.yaml")

state_path = model_dir / deploy_config["state_file"]

model_class = train_config["model_class"]
preprocess_config = model_config[model_class]["preprocess"]
body_config = model_config[model_class]["body"]
optim_config = model_config[model_class]["optim"]

# load pretrained language model
tokenizer = get_tokenizer(
    source=preprocess_config["tokenizer_source"],
    name=preprocess_config["tokenizer_name"],
)
pretrained_lm = None
pretrained_word_emb = None
word2idx = None
add_special_tokens = True

pretrained_lm = PretrainedLM(body_config["pretrained_lm"])
pretrained_lm.resize_token_embeddings(tokenizer=tokenizer)

model = init_model(
    model_class=model_class,
    body_config=body_config,
    num_labels=3,
    pretrained_emb=pretrained_word_emb,
    num_emb=len(word2idx) if word2idx else None,
    pretrained_lm=pretrained_lm,
    device=device,
    state_path=state_path,
)

model.eval()


class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg


class ModelRunner:
    def __init__(
        self,
        return_score=False,
        return_tgt_pool=False,
        return_tgt_mask=False,
        return_all_repr=False,
        return_attn=False,
    ):
        self.queue = []
        self.queue_lock = None
        self.needs_processing = None
        self.needs_processing_timer = None
        self.return_score = return_score
        self.return_tgt_pool = return_tgt_pool
        self.return_tgt_mask = return_tgt_mask
        self.return_all_repr = return_all_repr
        self.return_attn = return_attn

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
            # if len(self.queue) >= MAX_QUEUE_SIZE:
            #     raise HandlingError("I'm too busy", code=503)
            self.queue.append(our_task)
            logger.debug("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed()

        await our_task["done_event"].wait()
        return our_task["output"]

    def run_model(self, input):  # runs in other thread
        with torch.no_grad():
            outputs = dict()
            results = model(
                **input,
                return_tgt_pool=self.return_tgt_pool,
                return_tgt_mask=self.return_tgt_mask,
                return_all_repr=self.return_all_repr,
                return_attn=self.return_attn
            )

            outputs["sentiment_id"] = torch.argmax(results[1], dim=1)
            outputs["score"] = torch.nn.functional.softmax(results[1], dim=1)
            outputs["tgt_pool"] = results[2]
            outputs["tgt_mask"] = results[3]
            outputs["all_repr"] = results[4]
            outputs["attn"] = results[5]
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
            for col in model.INPUT_COLS:

                if col in to_process[0]["input"].features:
                    input[col] = torch.stack(
                        [t["input"].features[col] for t in to_process], dim=0
                    ).to(device)

            # we could delete inputs here...

            results = await app.loop.run_in_executor(
                None, functools.partial(self.run_model, input)
            )
            # result = self.run_model(input)

            results["sentiment"] = [
                SENTI_ID_MAP_INV[p]
                for p in results["sentiment_id"].detach().cpu().numpy()
            ]
            # if self.return_tgt_repr:
            results["tgt_pool"] = (
                results["tgt_pool"].detach().cpu().numpy()
                if results["tgt_pool"] is not None
                else None
            )
            results["tgt_mask"] = (
                results["tgt_mask"].detach().cpu().numpy()
                if results["tgt_mask"] is not None
                else None
            )
            results["all_repr"] = (
                results["all_repr"].detach().cpu().numpy()
                if results["all_repr"] is not None
                else None
            )
            # if self.return_attn:
            results["attn"] = (
                np.array([a.detach().cpu().numpy() for a in results["attn"]])
                if results["attn"] is not None
                else None
            )
            # if self.return_score:
            results["score"] = (
                results["score"].detach().cpu().numpy().max(axis=1)
                if results["score"] is not None
                else None
            )

            for i in range(len(to_process)):
                output = dict()
                t = to_process[i]
                output["sentiment_id"] = int(results["sentiment_id"][i])
                output["sentiment"] = results["sentiment"][i]
                output["tgt_pool"] = (
                    results["tgt_pool"][i].tolist()
                    if results["tgt_pool"] is not None
                    else None
                )
                output["tgt_mask"] = (
                    results["tgt_mask"][i].tolist()
                    if results["tgt_mask"] is not None
                    else None
                )
                output["all_repr"] = (
                    results["all_repr"][i].tolist()
                    if results["all_repr"] is not None
                    else None
                )
                output["attn"] = (
                    results["attn"][i].tolist() if results["attn"] is not None else None
                )
                output["score"] = (
                    float(results["score"][i]) if results["score"] is not None else None
                )
                t["output"] = output
                t["done_event"].set()
            del to_process


model_runner = ModelRunner(
    return_score=True,
    return_tgt_pool=False,
    return_tgt_mask=False,
    return_all_repr=False,
    return_attn=True,
)


@app.route("/target_sentiment", methods=["POST"], stream=True)
async def target_sentiment(request):
    try:

        body = await request.stream.read()
        body = json.loads(body)

        if "all_in_content_fmt" not in body or body["all_in_content_fmt"] == 0:
            body = parse_api_req(body)

        response = dict()
        example = TargetDependentExample(
            raw_text=body["content"],
            raw_start_idx=body["start_ind"],
            raw_end_idx=body["end_ind"],
            tokenizer=tokenizer,
            label=None,
            preprocess_config=preprocess_config,
            required_features=model.INPUT_COLS + ['target_span'],
        )
            

        results = await model_runner.process_input(example)
        response["sentiment"] = results["sentiment"]
        response["score"] = results["score"]

        if "explanation" in body:
            explanation_params = body["explanation_params"] if "explanation_params" in body else {}
            if body["explanation"]=="lime":
                expl = LimeExplanation(
                    self, 
                    model=model, 
                    tokenizer=tokenizer,
                    features=example.features,  
                    non_negative=True, 
                    exclude_spec_tokens=True, 
                    num_samples=explanation_params['num_samples'] if 'num_samples' in explanation_params else 500, 
                    faithfulness=explanation_params['faithfulness'] if 'faithfulness' in explanation_params else False
                )
                
            
            response['saliency_map'] = expl.scores
            response["sentiment"] = str(expl.sentiment)
            response["target_span"] = example.features['target_span'].tolist()
            response["target_tokens"] = example.features['target_tokens']
            response["score"] = float(expl.score)

        response["message"] = "OK"
        return sanic.response.json(response, status=200)

    except Exception as e:
        msg = traceback.format_exc()
        return sanic.response.json({"message": msg}, status=500)


app.add_task(model_runner.model_runner())
app.run(host="0.0.0.0", port=8080, debug=False)
