from torch.utils.data import DataLoader
from explain import get_explanation_model
from explain.faithfulness import Faithfulness
from tqdm import tqdm 
import logging 
import numpy as np


logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, model, dataset, args):
        self.args = args
        self.explain_config = self.args.explain_config
        print(self.explain_config)
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
            # explanations.extend(scores)
            sufficiency.extend(faithfulness.sufficiency)
            comprehensiveness.extend(faithfulness.comprehensiveness)

        # insert back to diagnosis df in dataset

        # print(sufficiency)
        sufficiency = np.array(sufficiency)
        # print(comprehensiveness)
        comprehensiveness = np.array(comprehensiveness)

        print(sufficiency.mean(axis=0))
        print(comprehensiveness.mean(axis=0))

        # return explanations, faithfulnesses


        



