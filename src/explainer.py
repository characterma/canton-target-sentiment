from torch.utils.data import DataLoader
from explain import get_explanation_model
from explain.faithfulness import Faithfulness
from tqdm import tqdm 
import logging 
import numpy as np
import pandas as pd 


logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, model, dataset, args):
        self.args = args
        self.explain_config = self.args.explain_config
        
        self.model = model
        self.dataset = dataset 

        self.explanation_model = get_explanation_model(
            model=model, 
            args=args
        )

    def explain(self):
        logger.info("***** Running explanation *****")
        logger.info("  Num examples = %d", len(self.dataset))

        dataloader = DataLoader(
            self.dataset,
            shuffle=False, 
            batch_size=self.explain_config["batch_size"],
        )
        
        explanations = []
        sufficiency = []
        comprehensiveness = []

        for idx, batch in tqdm(enumerate(dataloader)):
            scores = self.explanation_model(
                batch=batch, 
                target=None
            )

            faithfulness = Faithfulness(
                model=self.model, 
                batch=batch, 
                scores=scores, 
                mask_id=100,
                args=self.args
            )
            explanations.extend(scores.tolist())
            sufficiency.extend(faithfulness.sufficiency)
            comprehensiveness.extend(faithfulness.comprehensiveness)

        # explanations # [N, L]
        # sufficiency # list: [N, 5]
        # comprehensiveness # list: [N, 5]

        self.dataset.insert_diagnosis_column(explanations, "explanations")
        self.dataset.insert_diagnosis_column(sufficiency, "sufficiency")
        self.dataset.insert_diagnosis_column(comprehensiveness, "comprehensiveness")

        tokens = self.dataset.diagnosis_df['tokens'].tolist()
        tokens_sorted = []

        for t, s in zip(tokens, explanations):
            tkns = zip(t, s)
            tkns_sorted = sorted(tkns, key=lambda x: x[1], reverse=True)
            tokens_sorted.append(
                [t[0] for t in tkns_sorted]
            )

        self.dataset.insert_diagnosis_column(tokens_sorted, "tokens_sorted")

        sufficiency_avg = np.mean(sufficiency, axis=0)
        comprehensiveness_avg = np.mean(comprehensiveness, axis=0)
        faithfulness_rep = pd.DataFrame(
            data={
                "p": np.arange(0,6)/10, 
                "sufficiency_avg": sufficiency_avg, 
                "comprehensiveness_avg": comprehensiveness_avg}
    
        )
        faithfulness_rep.to_csv(
            self.args.result_dir / f"faithfulness_rep_{self.args.explain_config['model_class'].lower()}.csv", 
            index=False
        )







        



