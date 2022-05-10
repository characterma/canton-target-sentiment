import re
import random
import itertools
import numpy as np
from ailabuap.tokenizer import LTPTokenizer
from ailabuap.io.api_caller import APICaller
from utils import chunks


def uap_api_caller(url, batch_input):
    api_caller = APICaller(
        url=url,
        method="POST",
    )
    result_list = api_caller.call_batch_async(batch_input,)
    return result_list


def ner_tagging(ner_endpoint, texts):
    inputs = []
    for i, t in enumerate(texts):
        if t:
            t = str(t)
            test_data = dict(json={"docid": str(i), "text": t})
            inputs.append(test_data)
        else:
            continue

    if len(inputs) > 100:
        outputs = []
        iter_inputs = chunks(inputs, 100)
        for i, batch in enumerate(iter_inputs):
            batch_outpus = uap_api_caller(ner_endpoint, batch)
            outputs.extend(batch_outpus)
    else:
        outputs = uap_api_caller(ner_endpoint, inputs)

    return outputs.success


def extract_target_entity(text, target_type, url):
    entities = ner_tagging(url, [text])[0]["result"]
    tgt_entities = [rec for rec in entities if "label_name" in rec and rec["label_name"].lower() == target_type.lower()]
    return tgt_entities


def cut_text_spans(text, idxs):
    current_idx = 0
    spans = []
    for idx in idxs:
        kw_start = idx[0]
        kw_end = idx[1]

        if kw_start > current_idx:
            spans.append((text[current_idx: kw_start], "non-kw"))

        spans.append((text[kw_start: kw_end], "kw"))
        current_idx = kw_end

    if current_idx < len(text):
        spans.append((text[current_idx:], "non-kw"))

    return spans


class Perturbation:

    tokenizer = None

    @staticmethod
    def perturb(data, perturb_fn, nsamples=None, *args, **kwargs):

        augmented_data_dict = []
        augmented_data_inputs = []

        perturb_ratio = kwargs.get("purturbation_ratio")
        max_n_augs = 1 # TODO: adjust according to perturb_ratio

        for i, data_record in enumerate(data):

            text = data_record["content"] if "content" in data_record else data_record["text"]
            inv_text_inputs = [str(text)]
            inv_dict_inputs = [data_record]

            if len(text) == 0:
                continue

            if "text_subjs" in data_record:
                text_subjs = data_record["text_subjs"]
                kw_idxs = text_subjs["kw_idxs"]
                # text_idxs = text_subjs["text_idxs"]

                text_spans = cut_text_spans(text, kw_idxs[0])

                # apply perturbation on each span
                augmented_records = perturb_fn(text_spans, *args, **kwargs)

                # reconstruct text and update index
                if augmented_records:
                    for augmented_spans in augmented_records:

                        aug_record = Perturbation.reconstruct_data_and_kwidxs(augmented_spans)

                        if aug_record["n_augs"] == 0 or aug_record["n_augs"] > max_n_augs:
                            print(aug_record)
                            continue

                        new_data = {}
                        new_ct = aug_record["content"]
                        new_data["content"] = new_ct
                        new_data["docid"] = data_record.get("docid", "202202118983")
                        new_data["label"] = data_record.get("label")
                        new_data["text_subjs"] = text_subjs.copy()
                        new_data["text_subjs"]["name"] = aug_record["kw_name"]
                        new_data["text_subjs"]["kw_idxs"] = [aug_record["kw_idxs"]]
                        new_data["text_subjs"]["text_idxs"] = [[0, len(new_ct)]]

                        inv_text_inputs.append(new_ct)
                        inv_dict_inputs.append(new_data.copy())

            else:
                augmented_data = perturb_fn(text, *args, **kwargs)
                if augmented_data:
                    for new_text in augmented_data:
                        new_data_record = data_record.copy()
                        new_data_record["content"] = new_text

                        inv_text_inputs.append(new_text)
                        inv_dict_inputs.append(new_data_record)

            # construct the inputs for INV
            augmented_data_inputs.append(inv_text_inputs)
            augmented_data_dict.append(inv_dict_inputs)

        if nsamples and nsamples < len(augmented_data_inputs):
            order = list(range(len(augmented_data_inputs)))
            np.random.shuffle(order)
            order = order[:nsamples]

            augmented_data_inputs = [augmented_data_inputs[i] for i in order]
            augmented_data_dict = [augmented_data_dict[i] for i in order]

        return augmented_data_inputs, augmented_data_dict

    @staticmethod
    def _get_tokenizer():
        if not Perturbation.tokenizer:
            Perturbation.tokenizer = LTPTokenizer(mode=LTPTokenizer.mode.pos)
        return Perturbation.tokenizer

    @staticmethod
    def reconstruct_data_and_kwidxs(augmented_spans):
        text = ""
        kw_name = ""
        indexs = []
        start_idx = 0
        n_augs = 0
        for span in augmented_spans:
            span_text, span_type, n_changes = span

            text += span_text
            n_augs += n_changes
            if span_type == "kw":
                if kw_name == "":
                    kw_name = span_text
                indexs.append([start_idx, len(text)])
            start_idx = len(text)

        record = {
            "content": text,
            "kw_name": kw_name,
            "kw_idxs": indexs,
            "n_augs": n_augs
        }

        return record

    @staticmethod
    def apply_func_on_spans(func, data, skip_kw=True):
        if isinstance(data, str):
            results = func(data)
            results = [r[0] for r in results]
        elif isinstance(data, list):  # list of span
            augmented_spans = []
            for span in data:
                span_text, span_type = span
                if skip_kw and span_type == "kw":
                    span = span + (0,)
                    augmented_spans.append([span])
                else:
                    augmented_strings = func(span_text)
                    augmented_unit = [span + (0,)] + [(s[0], span_type, s[1]) for s in augmented_strings]
                    augmented_spans.append(augmented_unit)
            results = list(itertools.product(*augmented_spans))
        else:
            results = []

        return results

    @staticmethod
    def remove_random(data, purturbation_ratio=0.3, segmentation="word", n_samples=3, **kwargs):

        def _remove_random(text):
            results = []
            if segmentation == "word":
                tokenizer = Perturbation._get_tokenizer()
                _, segments = tokenizer(text)
                candidates = []
                tokens = []
                for i, seg_info in enumerate(segments):
                    word = seg_info["text"]
                    flag = seg_info["postag"]
                    tokens.append(word)
                    if flag not in ["a", "c", "d", "v", "wp", "z"]:
                        candidates.append(i)
            else:
                tokens = list(text)
                candidates = list(range(len(tokens)))

            n_total = len(candidates)
            n_to_rm = int(n_total * purturbation_ratio)

            if n_to_rm > 0:
                for _ in range(n_samples):
                    this_candidates = candidates.copy()

                    unused_idx = random.sample(this_candidates, n_to_rm)

                    # construct text
                    new_text = [w for i, w in enumerate(tokens) if i not in unused_idx]
                    new_text = "".join(new_text)
                    # results.append((new_text, len(unused_idx)))
                    results.append((new_text, 1))

            return results

        results = Perturbation.apply_func_on_spans(_remove_random, data, skip_kw=True)

        return results

    @staticmethod
    def remove_entity(data, api_url, target_type="company", n_samples=3, **kwargs):
        def _remove_entity(text):

            results = []
            target_entities = extract_target_entity(text, target_type, api_url)

            n_total = len(target_entities)
            sample_size = min(n_samples, n_total)

            if sample_size > 0:
                for _ in range(sample_size):
                    unused_entity = target_entities.pop(random.randrange(len(target_entities)))
                    start_index = unused_entity["start_ind"]
                    end_index = unused_entity["end_ind"]
                    new_text = text[:start_index] + text[end_index:]
                    results.append((new_text, 1))

            return results

        results = Perturbation.apply_func_on_spans(_remove_entity, data, skip_kw=True)

        return results

    @staticmethod
    def remove_keyword(data, keywords, n_samples=3, purturbation_ratio=0.5, **kwargs):
        def _remove_keyword(text):
            results = []

            re_keywords = [re.escape(kw) for kw in keywords]
            pattern = r"%s" % "|".join(re_keywords)

            if re.search(pattern, text):
                matches = list(re.finditer(pattern, text))
                n_total = len(matches)
                n_drop = max(1, int(purturbation_ratio * n_total))

                for _ in range(n_samples):
                    a = list(text)

                    items = random.sample(matches, n_drop)
                    for item in items:
                        start = item.start()
                        end = item.end()
                        a[start: end] = [""] * (end - start)

                    results.append(("".join(a), n_drop))

            results = list(set(results))

            return results

        results = Perturbation.apply_func_on_spans(_remove_keyword, data, skip_kw=True)

        return results

    @staticmethod
    def change_entity(data, api_url, alternatives, target_type="company", n_samples=3, **kwargs):
        def _change_entity(text):
            results = []
            target_entities = extract_target_entity(text, target_type, api_url)
            n_total = len(target_entities)

            if n_total > 0:
                sample_size = min(n_samples, len(alternatives))
                n_samples_per = sample_size // n_total

                if n_samples_per == 0:
                    random.shuffle(target_entities)
                    target_entities = target_entities[:sample_size]

                for target_entity in target_entities:

                    start_index = target_entity["start_ind"]
                    end_index = target_entity["end_ind"]
                    text_segment = target_entity["text_segment"]

                    if text_segment in alternatives:
                        alternatives.remove(text_segment)

                    random.shuffle(alternatives)
                    for i in range(n_samples_per):
                        replacement = alternatives[i]
                        sample = text[:start_index] + replacement + text[end_index:]
                        results.append((sample, 1))

            return results

        results = Perturbation.apply_func_on_spans(_change_entity, data, skip_kw=False)

        return results

    @staticmethod
    def change_dict(data, replacement_dict, n_samples=3, **kwargs):
        def _change_dict(text):

            results = []
            for k, v in replacement_dict.items():
                if re.search(r'%s' % k, text):
                    results.extend([(re.sub(r'%s' % k, vv, text), 1) for vv in v])

            if len(results) > n_samples:
                results = random.sample(results, n_samples)

            return results

        results = Perturbation.apply_func_on_spans(_change_dict, data, skip_kw=False)

        return results

    @staticmethod
    def change_keyword(data, keywords, n_samples=3, **kwargs):
        def _change_keyword(text):
            results = []
            for p in keywords:
                if re.search(r'%s' % re.escape(p), text):
                    results.extend([(re.sub(r'%s' % re.escape(p), p2, text), 1) for p2 in keywords if p != p2])

            if len(results) > n_samples:
                results = random.sample(results, n_samples)

            return results

        results = Perturbation.apply_func_on_spans(_change_keyword, data, skip_kw=False)

        return results


if __name__ == "__main__":
    # data = [
    #     {"content": "#苹果发布会# 说实话入耳的airpods pro 我带得不是很舒服 [泪]但是是大侠送的"},
    #     # {"content": "新的MacBook Pro键盘改实体了，可能苹果发现触控的不好用，哈哈哈哈哈[允悲]#苹果发布会#"},
    #     # {"content": "#苹果发布会#给我赶快发货！！！[怒]"}
    # ]

    data = [
        {'docid': '20220124A06Q2PC',
            'content': '粵財控股115億元拿下南粵銀行近六成股份,"入主"資格已獲監管批准',
            'text_subjs': {'id': '1',
                        'name': '南粵銀行',
                        'kw_idxs': [[[11, 15]]],
                        'text_idxs': [[0, 33]]},
            'label': -1}
    ]
    # results = Perturbation.perturb(data, Perturbation.random_deletion_without_target)
    results = Perturbation.perturb(
        data, Perturbation.remove_entity,
        api_url="http://ess25.wisers.com/playground/ner-kd-jigou-gpu/entityextr/analyse",
        target_type="company")
    print(results)
