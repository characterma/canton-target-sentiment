from torch.utils.data import DataLoader
from explain import ExplainModel
from explain.faithfulness import Faithfulness
from tqdm import tqdm 
import logging 
import numpy as np
import pandas as pd 


logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, model, args, run_faithfulness=True):
        self.args = args
        self.model = model
        self.config = self.args.explain_config
        self.device = args.device

        self.explanation_model = ExplainModel(
            method=self.config['method'], 
            model=self.model, 
            layer=self.config['layer'],
            logits_index=2
        )

        self.run_faithfulness = run_faithfulness

    def make_inputs(self, batch):
        inputs = dict()
        for col in batch:
            inputs[col] = batch[col].to(self.device)
        return inputs

    def explain(self, dataset):
        logger.info("***** Running explanation *****")
        logger.info("  Num examples = %d", len(dataset))

        dataloader = DataLoader(
            dataset,
            shuffle=False, 
            batch_size=self.config["batch_size"],
        )
        
        explanations = []
        sufficiency = []
        comprehensiveness = []

        for idx, batch in tqdm(enumerate(dataloader)):
            inputs = self.make_inputs(batch)
            scores = self.explanation_model(inputs=inputs)
            explanations.extend(scores.tolist())

            if self.run_faithfulness:
                faithfulness = Faithfulness(
                    model=self.model, 
                    inputs=inputs, 
                    scores=scores, 
                    mask_id=100,
                    args=self.args
                )
                sufficiency.extend(faithfulness.sufficiency)
                comprehensiveness.extend(faithfulness.comprehensiveness)

        dataset.insert_diagnosis_column(explanations, "explanations")
        if self.run_faithfulness:
            dataset.insert_diagnosis_column(sufficiency, "sufficiency")
            dataset.insert_diagnosis_column(comprehensiveness, "comprehensiveness")

        tokens = dataset.diagnosis_df['tokens'].tolist()
        tokens_sorted = []

        for t, s in zip(tokens, explanations):
            tkns = zip(t, s)
            tkns_sorted = sorted(tkns, key=lambda x: x[1], reverse=True)
            tokens_sorted.append(
                [t[0] for t in tkns_sorted]
            )

        dataset.insert_diagnosis_column(tokens_sorted, "tokens_sorted")
        
        if self.run_faithfulness:
            sufficiency_avg = np.mean(sufficiency, axis=0)
            comprehensiveness_avg = np.mean(comprehensiveness, axis=0)
            faithfulness_rep = pd.DataFrame(
                data={
                    "p": np.arange(0,6)/10, 
                    "sufficiency_avg": sufficiency_avg, 
                    "comprehensiveness_avg": comprehensiveness_avg}
        
            )
            faithfulness_rep.to_csv(
                self.args.result_dir / f"faithfulness_rep_{self.config['method'].lower()}.csv", 
                index=False
            )







        



