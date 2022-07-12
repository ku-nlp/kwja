import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import PretrainedConfig

from jula.models.models.cohesion_analyzer import CohesionAnalyzer, cohesion_cross_entropy_loss
from jula.models.models.dependency_parser import DependencyParser
from jula.utils.utils import IGNORE_INDEX


class RelationAnalyzer(nn.Module):
    def __init__(self, hparams: DictConfig, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()
        self.hparams = hparams

        self.dependency_parser: DependencyParser = DependencyParser(
            pretrained_model_config=pretrained_model_config,
            k=hparams.k,
        )
        self.cohesion_analyzer: CohesionAnalyzer = CohesionAnalyzer(
            pretrained_model_config=pretrained_model_config,
            num_rels=int("pas_analysis" in hparams.cohesion_tasks) * len(hparams.cases)
            + int("coreference" in hparams.cohesion_tasks)
            + int("bridging" in hparams.cohesion_tasks),
        )

    def forward(
        self,
        pooled_outputs: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = dict()
        # (b, seq, seq)
        dependency_logits, dependency_type_logits = self.dependency_parser(
            pooled_outputs,
            dependencies=batch["dependencies"] if batch["training"] else None,
        )
        dependency_logits = torch.where(
            batch["intra_mask"],
            dependency_logits,
            torch.full_like(dependency_logits, -1024.0),
        )
        output.update(
            {
                "dependency_logits": dependency_logits,
                "dependency_type_logits": dependency_type_logits,
            }
        )
        if "dependencies" in batch:
            dependency_loss = F.cross_entropy(
                input=dependency_logits.view(-1, dependency_logits.size(2)),
                target=batch["dependencies"].view(-1),
                ignore_index=IGNORE_INDEX,
            )
            output.update({"dependency_loss": dependency_loss})
        if "dependency_types" in batch:
            top1 = dependency_type_logits[:, :, 0, :]
            dependency_type_loss = F.cross_entropy(
                input=top1.view(-1, top1.size(2)),
                target=batch["dependency_types"].view(-1),
                ignore_index=IGNORE_INDEX,
            )
            output.update({"dependency_type_loss": dependency_type_loss})

        cohesion_logits = self.cohesion_analyzer(pooled_outputs)  # (b, rel, seq, seq)
        output.update(
            {
                "cohesion_logits": cohesion_logits,
            }
        )
        if "cohesion_target" in batch and "cohesion_mask" in batch:
            cohesion_loss = cohesion_cross_entropy_loss(
                cohesion_logits,
                batch["cohesion_target"],
                batch["cohesion_mask"],
            )
            output.update({"cohesion_loss": cohesion_loss})
        return output
