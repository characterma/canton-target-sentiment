# coding=utf-8
import argparse
import logging
from pathlib import Path
from trainer import Trainer, evaluate
from utils import (
    set_seed,
    set_log_path,
    load_config,
    save_config,
    log_args,
    get_args,
)
from utils import combine_and_save_metrics, combine_and_save_statistics
from dataset import get_dataset
from tokenizer import get_tokenizer
from model import get_model
from label import get_label_to_id
from trainer_kd import get_logits, KDTrainer
from explainer import Explainer


logger = logging.getLogger(__name__)


def run_kd(args):
    # load teacher
    teacher_dir = Path(args.kd_config["teacher_dir"])
    teacher_args = get_args(config_dir=teacher_dir)
    teacher_args = load_config(args=teacher_args)
    teacher_args.data_config = args.data_config
    teacher_tokenizer = get_tokenizer(args=teacher_args)
    teacher_args.label_to_id, teacher_args.label_to_id_inv = get_label_to_id(
        tokenizer=teacher_tokenizer, args=teacher_args
    )
    teacher_model = get_model(args=teacher_args)

    # load unlabeled data, which will be preprocessed by teacher pipeline TODO: allow label=null
    unlabeled_dataset = get_dataset(
        dataset="unlabeled", tokenizer=teacher_tokenizer, args=teacher_args
    )
    train_dataset = get_dataset(
        dataset="train", tokenizer=teacher_tokenizer, args=teacher_args
    )
    # TODO: include logits from train data

    # load student
    student_tokenizer = get_tokenizer(args=args, datasets=["train", "unlabeled"])
    args.label_to_id, args.label_to_id_inv = get_label_to_id(
        tokenizer=student_tokenizer, args=args
    )
    student_model = get_model(args=args)

    # generate soft-labels, TODO: cache to disk
    teacher_logits_ul = get_logits(
        model=teacher_model,
        dataset=unlabeled_dataset,
        teacher_args=teacher_args,
        student_args=args,
    )
    teacher_logits_tr = get_logits(
        model=teacher_model,
        dataset=train_dataset,
        teacher_args=teacher_args,
        student_args=args,
    )

    # Features for student model
    train_dataset = get_dataset(dataset="train", tokenizer=student_tokenizer, args=args)
    dev_dataset = get_dataset(dataset="dev", tokenizer=student_tokenizer, args=args)
    test_dataset = get_dataset(dataset="test", tokenizer=student_tokenizer, args=args)
    unlabeled_dataset = get_dataset(
        dataset="unlabeled", tokenizer=student_tokenizer, args=args
    )

    # merge teacher_logits into features
    train_dataset.add_feature("teacher_logit", teacher_logits_tr)
    unlabeled_dataset.add_feature("teacher_logit", teacher_logits_ul)

    # run kd_trainer => save model
    kd_trainer = KDTrainer(
        model=student_model,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        unlabeled_dataset=unlabeled_dataset,
        args=args,
    )

    kd_trainer.train()

    # evaluation
    train_metrics = evaluate(model=student_model, eval_dataset=train_dataset, args=args)
    dev_metrics = evaluate(model=student_model, eval_dataset=dev_dataset, args=args)
    test_metrics = evaluate(model=student_model, eval_dataset=test_dataset, args=args)

    combine_and_save_metrics(
        metrics=[train_metrics, dev_metrics, test_metrics], args=args
    )
    combine_and_save_statistics(
        datasets=[train_dataset, dev_dataset, test_dataset], args=args
    )
    save_config(args)


def run(args):
    if not args.test_only:
        save_config(args)

    tokenizer = get_tokenizer(args=args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    model = get_model(args=args)
    if not args.test_only:
        train_dataset = get_dataset(dataset="train", tokenizer=tokenizer, args=args)
        dev_dataset = get_dataset(dataset="dev", tokenizer=tokenizer, args=args)
        trainer = Trainer(
            model=model, train_dataset=train_dataset, dev_dataset=dev_dataset, args=args
        )
        trainer.train()
    else:
        train_dataset = None
        dev_dataset = None

    test_dataset = get_dataset(dataset="test", tokenizer=tokenizer, args=args)

    if not args.test_only:
        train_metrics = evaluate(model=model, eval_dataset=train_dataset, args=args)
        dev_metrics = evaluate(model=model, eval_dataset=dev_dataset, args=args)
    else:
        train_metrics = None
        dev_metrics = None

    if args.explain:
        if args.faithfulness:
            explainer = Explainer(
                model=model, 
                args=args, 
                run_faithfulness=True
            )
        else:
            explainer = Explainer(
                model=model, 
                args=args, 
                run_faithfulness=False
            )
        explainer.explain(dataset=test_dataset)

    test_metrics = evaluate(model=model, eval_dataset=test_dataset, args=args)
    combine_and_save_metrics(
        metrics=[train_metrics, dev_metrics, test_metrics], args=args
    )
    combine_and_save_statistics(
        datasets=[train_dataset, dev_dataset, test_dataset], args=args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../config/")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--faithfulness", action="store_true")
    args = parser.parse_args()

    args = load_config(args)
    set_log_path(args.output_dir)
    log_args(logger, args)
    set_seed(args.train_config["seed"])

    if not args.test_only and args.kd_config["use_kd"]:
        run_kd(args=args)
    else:
        run(args=args)
