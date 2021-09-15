from torch.utils.data import DataLoader
from explain import ExplainModel
from explain.faithfulness import Faithfulness
from tqdm import tqdm
import logging
import json
import numpy as np
import pandas as pd
import pickle as pkl


logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, model, args, run_faithfulness=True):
        self.args = args
        self.model = model
        self.config = self.args.explain_config
        self.device = args.device
        self.run_faithfulness = run_faithfulness

    def make_inputs(self, batch):
        inputs = dict()
        for col in batch:
            inputs[col] = batch[col].to(self.device)
        return inputs

    def explain(self, dataset):

        for idx, config in self.config.items():
            logger.info(f"***** Running explanation {idx} *****")
            logger.info("  Method = %s", config['method'])
            logger.info("  Num examples = %d", len(dataset))
            try:
                explanations = []
                sufficiency = []
                comprehensiveness = []
                decision_flip_mit = []
                decision_flip_fot = []
                importance_probability_correlation = []
                monotonicity = []
                masked_scores = []
                dataloader = DataLoader(
                    dataset, shuffle=False, batch_size=config.get("batch_size", 1)
                )
                explanation_model = ExplainModel(
                    model=self.model,
                    config=config,
                )

                for batch in tqdm(dataloader):
                    inputs = self.make_inputs(batch)
                    scores = explanation_model(inputs=inputs)
                    explanations.extend(scores.tolist())

                    if self.run_faithfulness:
                        mask_or_remove = config.get('mask_or_remove', 'remove')
                        unk_token_id = dataset.tokenizer.unk_token_id
                        pad_token_id = dataset.tokenizer.pad_token_id

                        faithfulness = Faithfulness(
                            model=self.model,
                            inputs=inputs,
                            scores=scores,
                            mask_or_remove=mask_or_remove,
                            unk_token_id=unk_token_id,
                            pad_token_id=pad_token_id
                        )
                        masked_scores.extend(faithfulness.masked_scores)
                        sufficiency.extend(faithfulness.sufficiency)
                        comprehensiveness.extend(faithfulness.comprehensiveness)
                        decision_flip_mit.extend(faithfulness.decision_flip_mit)
                        decision_flip_fot.extend(faithfulness.decision_flip_fot)
                        importance_probability_correlation.extend(faithfulness.importance_probability_correlation)
                        monotonicity.extend(faithfulness.monotonicity)

                dataset.insert_diagnosis_column(explanations, f"explanations_{idx}", update=True)
                if len(masked_scores) > 0:
                    pkl.dump(
                        masked_scores, 
                        open(self.args.result_dir / f"masked_scores_{idx}.pkl", "wb")
                    )

                if self.run_faithfulness:
                    dataset.insert_diagnosis_column(sufficiency, f"sufficiency_{idx}", update=True)
                    dataset.insert_diagnosis_column(comprehensiveness, f"comprehensiveness_{idx}", update=True)
                    dataset.insert_diagnosis_column(decision_flip_mit, f"decision_flip_mit_{idx}", update=True)
                    dataset.insert_diagnosis_column(decision_flip_fot, f"decision_flip_fot_{idx}", update=True)
                    dataset.insert_diagnosis_column(importance_probability_correlation, f"importance_probability_correlation_{idx}", update=True)
                    dataset.insert_diagnosis_column(monotonicity, f"monotonicity_{idx}", update=True)

                tokens = dataset.diagnosis_df["tokens"].tolist()
                tokens_sorted = []
                indice_sorted = []

                for t, s in zip(tokens, explanations):
                    tkns = zip(t, s)
                    idxs = zip(range(len(t)), s)
                    tkns_sorted = sorted(tkns, key=lambda x: x[1], reverse=True)
                    idxs_sorted = sorted(idxs, key=lambda x: x[1], reverse=True)
                    tokens_sorted.append([t[0] for t in tkns_sorted])
                    indice_sorted.append([t[0] for t in idxs_sorted])

                dataset.insert_diagnosis_column(tokens_sorted, f"tokens_sorted_{idx}", update=True)
                dataset.insert_diagnosis_column(indice_sorted, f"indice_sorted_{idx}", update=True)

                # dataset.diagnosis_df.to_excel(self.args.result_dir / f"explanations_{idx}.xlsx")

                if self.run_faithfulness:
                    sufficiency_avg = np.mean(sufficiency, axis=0)
                    comprehensiveness_avg = np.mean(comprehensiveness, axis=0)
                    faithfulness_rep = pd.DataFrame(
                        data={
                            "p": np.arange(1, 11) / 10,
                            "sufficiency": sufficiency_avg,
                            "comprehensiveness": comprehensiveness_avg,
                        }
                    )

                    importance_probability_correlation = [x for x in importance_probability_correlation if not np.isnan(x)]
                    monotonicity = [x for x in monotonicity if not np.isnan(x)]
                    faithfulness_sum = {
                        "decision_flip_mit": np.mean(decision_flip_mit, axis=0), 
                        "decision_flip_fot": np.mean(decision_flip_fot, axis=0), 
                        "sufficiency": np.mean(sufficiency_avg, axis=0),
                        "comprehensiveness": np.mean(comprehensiveness_avg, axis=0),
                        "importance_probability_correlation": np.mean(importance_probability_correlation, axis=0),
                        "monotonicity": np.mean(monotonicity, axis=0),
                    }
                    faithfulness_rep.to_csv(
                        self.args.result_dir 
                        / f"faithfulness_by_bins_{idx}.csv",
                        index=False,
                    )

                    json.dump(
                        faithfulness_sum, 
                        open(self.args.result_dir / f"faithfulness_summary_{idx}.json", "w")
                    )
                logger.info("  Finished.")
            except Exception as e:
                logger.info("  Failed: %s", e)
