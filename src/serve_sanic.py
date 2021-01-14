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

# we only run 1 inference run at any time (one could schedule between several runners if desired)
# MAX_QUEUE_SIZE = 64  # we accept a backlog of MAX_QUEUE_SIZE before handing out "Too busy" errors
MAX_BATCH_SIZE = 32  # we put at most MAX_BATCH_SIZE things in a single batch
MAX_WAIT = 0.1        # we wait at most MAX_WAIT seconds before running for more inputs to arrive in batching

import logging
from pathlib import Path
import os, sys
import pandas as pd
from transformers_utils import PretrainedLM
from utils import load_yaml, parse_api_req, SENTI_ID_MAP_INV
from tokenizer import get_tokenizer
from model import *
from dataset import TargetDependentExample
import traceback
import torch

# if len(sys.argv) > 1:
#     if sys.argv[1].isdigit():
#         device = int(sys.argv[1])
#     else:
#         device = sys.argv[1]
# else:
#     device = 1

device = 'cpu'

def init_model(
        model_class, 
        body_config, 
        num_labels, 
        pretrained_emb, 
        num_emb, 
        pretrained_lm, 
        device, 
        state_path=None
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

base_dir = Path("./")
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
tokenizer = get_tokenizer(source=preprocess_config["tokenizer_source"], name=preprocess_config["tokenizer_name"])
pretrained_lm = None
pretrained_word_emb = None
word2idx = None
add_special_tokens = True

pretrained_lm = PretrainedLM(body_config["pretrained_lm"])
pretrained_lm.resize_token_embeddings(tokenizer=tokenizer)

model = init_model(model_class=model_class, 
    body_config=body_config,
    num_labels=3, 
    pretrained_emb=pretrained_word_emb, 
    num_emb=len(word2idx) if word2idx else None, 
    pretrained_lm=pretrained_lm, 
    device=device, 
    state_path=state_path
)

model.eval()


class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg

class ModelRunner:
    def __init__(self):
        self.queue = []
        self.queue_lock = None
        self.needs_processing = None
        self.needs_processing_timer = None

    def schedule_processing_if_needed(self):
        if len(self.queue) >= MAX_BATCH_SIZE:
            logger.debug("next batch ready when processing a batch")
            self.needs_processing.set()
        elif self.queue:
            logger.debug("queue nonempty when processing a batch, setting next timer")
            self.needs_processing_timer = app.loop.call_at(self.queue[0]["time"] + MAX_WAIT, self.needs_processing.set)

    async def process_input(self, input):
        our_task = {"done_event": asyncio.Event(loop=app.loop),
                    "input": input,
                    "time": app.loop.time()}
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
            output = model(**input, return_reps=False)
            prediction = torch.argmax(output[1], dim=1)
            return prediction

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
                    logger.debug("launching processing. queue size: {}. longest wait: {}".format(len(self.queue), longest_wait))
                else:  # oops
                    longest_wait = None
                    logger.debug("launching processing. queue size: {}. longest wait: {}".format(len(self.queue), longest_wait))
                    continue
                
                to_process = self.queue[:MAX_BATCH_SIZE]
                del self.queue[:len(to_process)]
                self.schedule_processing_if_needed()
            # so here we copy, it would be neater to avoid this

            input = dict()
            for col in model.INPUT_COLS:

                if col in to_process[0]["input"].features:
                    input[col] = torch.stack(
                            [t["input"].features[col] for t in to_process], 
                            dim=0
                        ).to(device)

            # we could delete inputs here...

            result = await app.loop.run_in_executor(
                None, functools.partial(self.run_model, input)
            )
            # result = self.run_model(input)

            result = [SENTI_ID_MAP_INV[p] for p in result.detach().cpu().numpy()]
          
            for t, r in zip(to_process, result):
                t["output"] = r
                t["done_event"].set()
            del to_process

model_runner = ModelRunner()

@app.route('/target_sentiment', methods=['POST'], stream=True)
async def target_sentiment(request):
    try:

        body = await request.stream.read()
        body = json.loads(body)
        body = parse_api_req(body)
        # print(body)

        e = TargetDependentExample(
            raw_text=body["content"],
            raw_start_idx=body["start_ind"],
            raw_end_idx=body["end_ind"],
            tokenizer=tokenizer, 
            label=None,
            preprocess_config=preprocess_config,
            required_features=model.INPUT_COLS
        )

        output = await model_runner.process_input(e)
        return sanic.response.json({"data": output, "message":"OK"}, status=200)

    except Exception as e:
        msg = traceback.format_exc()
        return sanic.response.json({"message": msg}, status=500)

app.add_task(model_runner.model_runner())
app.run(host="0.0.0.0", port=8080, debug=False)