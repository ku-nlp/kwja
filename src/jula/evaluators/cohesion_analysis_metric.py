import numpy as np
import torch
from torchmetrics import Metric

from jula.datamodule.datasets.word_dataset import WordDataset
from jula.datamodule.examples import CohesionExample, Task
from jula.evaluators.cohesion_scorer import Scorer, ScoreResult
from jula.writer.knp import CohesionKNPWriter


class CohesionAnalysisMetric(Metric):
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        # Metric state variables can either be torch.Tensor or an empty list which can be used to store torch.Tensors`.
        # i.e. Expected metric state to be either a Tensor or a list of Tensor
        self.add_state("example_ids", default=list())  # list[torch.Tensor]
        self.add_state(
            "predictions", default=list()
        )  # list[torch.Tensor]  # [(rel, phrase)]

    def update(
        self,
        example_ids: torch.Tensor,  # (b)
        output: torch.Tensor,  # (b, rel, seq, seq)
        dataset: WordDataset,
    ) -> None:
        assert len(output) == len(example_ids)
        for out, eid in zip(output, example_ids):
            self.example_ids.append(eid)
            gold_example: CohesionExample = dataset.examples[eid.item()]
            # (rel, phrase, 0 or phrase+special)
            preds: list[list[list[float]]] = dataset.dump_prediction(
                out.tolist(), gold_example
            )
            self.predictions.append(
                torch.as_tensor(
                    [
                        [(np.argmax(ps).item() if ps else -1) for ps in pred]
                        for pred in preds
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
            )

    def compute(self, dataset: WordDataset) -> ScoreResult:
        knp_writer = CohesionKNPWriter(dataset)
        assert len(self.example_ids) == len(self.predictions)
        predictions = {
            eid.item(): prediction.tolist()
            for eid, prediction in zip(self.example_ids, self.predictions)
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
            dataset.documents,
            target_cases=dataset.cases,
            exophora_referents=dataset.exophora_referents,
            coreference=(Task.COREFERENCE in dataset.cohesion_tasks),
            bridging=(Task.BRIDGING in dataset.cohesion_tasks),
            pas_target=targets2label[tuple(dataset.pas_targets)],
        )
        score_result: ScoreResult = scorer.run()

        return score_result