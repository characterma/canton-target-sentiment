# coding=utf-8
import argparse
import logging
from pathlib import Path
import json
import os

from nlp_pipeline.trainer import Trainer, evaluate
from nlp_pipeline.trainer_uda import TrainerUDA
from nlp_pipeline.utils import (
    set_seed,
    set_log_path,
    load_config,
    save_config,
    log_args,
    get_args,
)
from nlp_pipeline.utils import combine_and_save_metrics, combine_and_save_statistics, start_mlops_log, stop_mlops_log
from nlp_pipeline.dataset import get_dataset
from nlp_pipeline.tokenizer import get_tokenizer
from nlp_pipeline.model import get_model
from nlp_pipeline.label import get_label_to_id
from nlp_pipeline.trainer_kd import get_logits, KDTrainer
from nlp_pipeline.explainer import Explainer


logger = logging.getLogger(__name__)


def run_uda(args):
    if not args.test_only:
        save_config(args)
    print(args.model_dir, "*****")
    tokenizer = get_tokenizer(args=args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    model = get_model(args=args)

    labeled_dataset = get_dataset(dataset="train", tokenizer=tokenizer, args=args)
    unlabeled_dataset = get_dataset(dataset="unlabeled", tokenizer=tokenizer, args=args)
    dev_dataset = get_dataset(dataset="dev", tokenizer=tokenizer, args=args)

    trainer = TrainerUDA(
        model=model, 
        labeled_dataset=labeled_dataset, 
        unlabeled_dataset=unlabeled_dataset, 
        dev_dataset=dev_dataset, 
        args=args
    )

    trainer.train()
    test_dataset = get_dataset(dataset="test", tokenizer=tokenizer, args=args)

    train_metrics = evaluate(model=model, eval_dataset=labeled_dataset, args=args)
    dev_metrics = evaluate(model=model, eval_dataset=dev_dataset, args=args)
    test_metrics = evaluate(model=model, eval_dataset=test_dataset, args=args)
    start_mlops_log(args=args)
    combine_and_save_metrics(
        metrics=[train_metrics, dev_metrics, test_metrics], args=args, suffix=args.suffix
    )
    combine_and_save_statistics(
        datasets=[labeled_dataset, dev_dataset, test_dataset], args=args, suffix=args.suffix
    )
    stop_mlops_log(args=args)


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

    # load student
    if "unlabeled" in args.data_config.keys() and args.data_config['unlabeled'] is not None:
        student_tokenizer = get_tokenizer(args=args, datasets=["train", "unlabeled"])
    else:
        student_tokenizer = get_tokenizer(args=args, datasets=["train"])
        
    args.label_to_id, args.label_to_id_inv = get_label_to_id(
        tokenizer=student_tokenizer, args=args
    )
    
    if "unlabeled" in args.data_config.keys() and args.data_config['unlabeled'] is not None:
        unlabeled_dataset = get_dataset(
            dataset="unlabeled", 
            tokenizer=teacher_tokenizer, 
            args=teacher_args
        )
    else:
        unlabeled_dataset = None 


    assert(teacher_args.label_to_id==args.label_to_id)
    assert(teacher_args.label_to_id_inv==args.label_to_id_inv)
    # same tokenizer can skip redundant loading of train and unlabel dataset
    same_tokenizer = teacher_args.model_config["tokenizer_source"]  == args.model_config["tokenizer_source"] and \
                        teacher_args.model_config["tokenizer_name"]  == args.model_config["tokenizer_name"]

    # generate soft-labels, TODO: cache to disk
    if unlabeled_dataset is not None:
        teacher_logits_ul = get_logits(
            model=teacher_model,
            dataset=unlabeled_dataset,
            teacher_args=teacher_args,
            student_args=args,
        )
        if not same_tokenizer:
            del unlabeled_dataset
    else:
        teacher_logits_ul = None 

    train_dataset = get_dataset(
        dataset="train", 
        tokenizer=teacher_tokenizer, 
        args=teacher_args
    )
    
    teacher_logits_tr = get_logits(
        model=teacher_model,
        dataset=train_dataset,
        teacher_args=teacher_args,
        student_args=args,
    )
    if not same_tokenizer:
        del train_dataset
    
    del teacher_model
    del teacher_tokenizer

    # Features for student model
    if not same_tokenizer:
        train_dataset = get_dataset(dataset="train", tokenizer=student_tokenizer, args=args)
    dev_dataset = get_dataset(dataset="dev", tokenizer=student_tokenizer, args=args)
    test_dataset = get_dataset(dataset="test", tokenizer=student_tokenizer, args=args)
    train_dataset.add_feature("teacher_logit", teacher_logits_tr)
    del teacher_logits_tr
    
    if "unlabeled" in args.data_config.keys() and args.data_config['unlabeled'] is not None:
        if not same_tokenizer:
            unlabeled_dataset = get_dataset(dataset="unlabeled", tokenizer=student_tokenizer, args=args)
        unlabeled_dataset.add_feature("teacher_logit", teacher_logits_ul)
        del teacher_logits_ul
    else:
        unlabeled_dataset = None

    # run kd_trainer => save model
    student_model = get_model(args=args)
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
    start_mlops_log(args=args)
    combine_and_save_metrics(
        metrics=[train_metrics, dev_metrics, test_metrics], args=args
    )
    combine_and_save_statistics(
        datasets=[train_dataset, dev_dataset, test_dataset], args=args
    )
    stop_mlops_log(args=args)
    save_config(args)


def run(args):
    import os
    if not args.test_only:
        save_config(args)
    print(args.model_dir, "*****")
    tokenizer = get_tokenizer(args=args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    model = get_model(args=args)
    if not args.test_only:
        train_dataset = get_dataset(dataset="train", tokenizer=tokenizer, args=args)
        dev_dataset = get_dataset(dataset="dev", tokenizer=tokenizer, args=args)
        additional_train_datasets = []
        if args.data_config.get("additional_train_dir"):
            for file_name in os.listdir(args.data_config["additional_train_dir"]):
                with open(os.path.join(args.data_config['additional_train_dir'], file_name), "rb") as f:
                    raw_data = json.load(f)
                additional_train_datasets.append(get_dataset(dataset="train", tokenizer=tokenizer, args=args, raw_data=raw_data))
        trainer = Trainer(
            model=model, train_dataset=train_dataset, dev_dataset=dev_dataset, args=args,
            additional_train_datasets=additional_train_datasets)
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
    start_mlops_log(args=args)
    combine_and_save_metrics(
        metrics=[train_metrics, dev_metrics, test_metrics], args=args, suffix=args.suffix
    )
    combine_and_save_statistics(
        datasets=[train_dataset, dev_dataset, test_dataset], args=args, suffix=args.suffix
    )
    stop_mlops_log(args=args)
    
    if args.al_config.get("use_al") and not args.test_only:
        logger.info("***** Running active learning *****")
        from nlp_pipeline.active_learning import query_active_learning_data
        import os
        import pandas as pd

        al_dataset = get_dataset(dataset="al_unlabel", tokenizer=tokenizer, args=args)
        _ = evaluate(model=model, eval_dataset=al_dataset, args=args, get_embeddings=args.al_config["query_method"] in ["cal", "coreset", "hybrid"])
        
        # Filter out already labeled data
        labeled_docid = list(train_dataset.diagnosis_df['docid']) + list(dev_dataset.diagnosis_df['docid']) + list(test_dataset.diagnosis_df['docid'])
        for dataset in additional_train_datasets:
            labeled_docid += list(dataset.diagnosis_df['docid'])
        
        query_data = query_active_learning_data(al_dataset.diagnosis_df, args=args, labeled_docid=labeled_docid)

        # Save queried data 
        with open(os.path.join(args.data_dir, args.data_config["al_unlabel"]), "rb") as f:
            unlabel_data = pd.DataFrame(json.load(f))
        query_data_df = unlabel_data[unlabel_data['docid'].isin(set(query_data['docid']))]
        iter_ind = 0
        logger.info("***** Saving active learning data *****")
        while True:
            file_name = f"{args.al_config.get('output_file', 'active_learning_data')}_{iter_ind}.json"
            if file_name not in os.listdir(args.al_config["output_dir"]):
                query_data_df.to_json(os.path.join(args.al_config["output_dir"], file_name), orient='records')
                break
            
            iter_ind += 1


def run_al_exp(args):
    import os
    import pickle
    
    print(args.model_dir, "*****")
    tokenizer = get_tokenizer(args=args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    train_dataset = get_dataset(dataset="train", tokenizer=tokenizer, args=args)
    dev_dataset = get_dataset(dataset="dev", tokenizer=tokenizer, args=args)
    test_dataset = get_dataset(dataset="test", tokenizer=tokenizer, args=args)
    if args.al_config.get("use_al") and not args.test_only:
        al_dataset = get_dataset(dataset="al_unlabel", tokenizer=tokenizer, args=args)

    iteration_result = []
    for i in range(args.al_config.get("iteration")):
        save_config(args)

        model = get_model(args=args)
        additional_train_datasets = []
        if args.data_config.get("additional_train_dir"):
            for file_name in os.listdir(args.data_config["additional_train_dir"]):
                with open(os.path.join(args.data_config['additional_train_dir'], file_name), "rb") as f:
                    raw_data = json.load(f)
                additional_train_datasets.append(get_dataset(dataset="train", tokenizer=tokenizer, args=args, raw_data=raw_data))
        trainer = Trainer(
            model=model, train_dataset=train_dataset, dev_dataset=dev_dataset, args=args,
            additional_train_datasets=additional_train_datasets)
        trainer.train()

        train_metrics = evaluate(model=model, eval_dataset=train_dataset, args=args)
        dev_metrics = evaluate(model=model, eval_dataset=dev_dataset, args=args)
        test_metrics = evaluate(model=model, eval_dataset=test_dataset, args=args)

        iteration_result.append(test_metrics)

        with open(args.al_config.get("result_file", "../active_learning_result/active_learning_result.pkl"), "wb") as f:
            pickle.dump(iteration_result, f)

        start_mlops_log(args=args)
        combine_and_save_metrics(
            metrics=[train_metrics, dev_metrics, test_metrics], args=args, suffix=args.suffix
        )
        combine_and_save_statistics(
            datasets=[train_dataset, dev_dataset, test_dataset], args=args, suffix=args.suffix
        )
        stop_mlops_log(args=args)

        if args.al_config.get("use_al") and not args.test_only:
            logger.info("***** Running active learning *****")
            from nlp_pipeline.active_learning import query_active_learning_data
            import os
            import pandas as pd

            _ = evaluate(model=model, eval_dataset=al_dataset, args=args, get_embeddings=args.al_config["query_method"] in ["cal", "coreset", "hybrid"])
            
            # Filter out already labeled data
            labeled_docid = list(train_dataset.diagnosis_df['docid']) + list(dev_dataset.diagnosis_df['docid']) + list(test_dataset.diagnosis_df['docid'])
            for dataset in additional_train_datasets:
                labeled_docid += list(dataset.diagnosis_df['docid'])

            query_data = query_active_learning_data(al_dataset.diagnosis_df, args=args, labeled_docid=labeled_docid)

            # Save queried data 
            with open(os.path.join(args.data_dir, args.data_config["al_unlabel"]), "rb") as f:
                unlabel_data = pd.DataFrame(json.load(f))
            query_data_df = unlabel_data[unlabel_data['docid'].isin(set(query_data['docid']))]
            iter_ind = 0
            logger.info("***** Saving active learning data *****")
            while True:
                file_name = f"{args.al_config.get('output_file', 'active_learning_data')}_{iter_ind}.json"
                if file_name not in os.listdir(args.al_config["output_dir"]):
                    query_data_df.to_json(os.path.join(args.al_config["output_dir"], file_name), orient='records')
                    break
                
                iter_ind += 1
        
        model_dir = args.output_dir
        os.remove(model_dir / "model/model.pt")
        
        del trainer
        del model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../config/")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--faithfulness", action="store_true")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    args = load_config(args)
    set_log_path(args.output_dir)
    log_args(logger, args)
    set_seed(args.train_config["seed"])

    if not args.test_only and args.kd_config["use_kd"]:
        run_kd(args=args)
    elif not args.test_only and args.uda_config["use_uda"]:
        run_uda(args=args)
    elif not args.test_only and args.al_config["run_al_exp"]:
        run_al_exp(args=args)
    else:
        run(args=args)
