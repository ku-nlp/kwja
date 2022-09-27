from statistics import mean

import numpy as np
import torch
from torchmetrics import Metric

from kwja.datamodule.datasets import WordDataset
from kwja.datamodule.examples import CohesionExample, CohesionTask
from kwja.evaluators.cohesion_scorer import Scorer, ScoreResult
from kwja.utils.cohesion import CohesionKNPWriter


class CohesionAnalysisMetric(Metric):
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        # Metric state variables can either be torch.Tensor or an empty list which can be used to store torch.Tensors`.
        # i.e. Expected metric state to be either a Tensor or a list of Tensor
        self.add_state("example_ids", default=list())
        self.add_state("predictions", default=list())
        self.example_ids: list[torch.Tensor]  # [()]
        self.predictions: list[torch.Tensor]  # [(rel, phrase)]

    def update(
        self,
        example_ids: torch.Tensor,  # (b)
        output: torch.Tensor,  # (b, rel, seq, seq)
        dataset: WordDataset,
    ) -> None:
        assert len(output) == len(example_ids)
        for out, eid in zip(output, example_ids):
            self.example_ids.append(eid)
            gold_example: CohesionExample = dataset.examples[eid.item()].cohesion_example
            # (rel, phrase, 0 or phrase+special)
            preds: list[list[list[float]]] = dataset.dump_prediction(out.tolist(), gold_example)
            self.predictions.append(
                torch.as_tensor(
                    [[(np.argmax(ps).item() if ps else -1) for ps in pred] for pred in preds],
                    dtype=torch.long,
                    device=example_ids.device,
                )
            )

    def compute(self, dataset: WordDataset) -> tuple[ScoreResult, dict[str, float]]:
        knp_writer = CohesionKNPWriter(dataset)
        assert len(self.example_ids) == len(self.predictions), f"{len(self.example_ids)} vs {len(self.predictions)}"
        predictions: dict[int, list[list[int]]] = {
            eid.item(): prediction.tolist() for eid, prediction in zip(self.example_ids, self.predictions)
        }
        documents_pred = knp_writer.write(predictions, destination=None)
        targets2label = {
            tuple(): "",
            ("pred",): "pred",
            ("noun",): "noun",
            ("pred", "noun"): "all",
        }

        scorer = Scorer(
            documents_pred,
            dataset.orig_documents,
            target_cases=dataset.pas_cases,
            exophora_referents=dataset.exophora_referents,
            coreference=(CohesionTask.COREFERENCE in dataset.cohesion_tasks),
            bridging=(CohesionTask.BRIDGING in dataset.cohesion_tasks),
            pas_target=targets2label[tuple(dataset.extractors[CohesionTask.PAS_ANALYSIS].pas_targets)],
        )
        score_result: ScoreResult = scorer.run()
        ret_dict = {}
        for rel, val in score_result.to_dict().items():
            for met, sub_val in val.items():
                ret_dict[f"{met}_{rel}"] = sub_val.f1
        f1s = []
        if CohesionTask.PAS_ANALYSIS in dataset.cohesion_tasks:
            f1s.append(ret_dict["pas_all_case"])
        if CohesionTask.BRIDGING in dataset.cohesion_tasks:
            f1s.append(ret_dict["bridging_all_case"])
        if CohesionTask.COREFERENCE in dataset.cohesion_tasks:
            f1s.append(ret_dict["coreference_all_case"])
        ret_dict["cohesion_analysis_f1"] = mean(f1s)

        return score_result, ret_dict
