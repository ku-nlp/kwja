import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import PretrainedConfig

from kwja.models.models.cohesion_analyzer import CohesionAnalyzer, cohesion_cross_entropy_loss
from kwja.models.models.dependency_parser import DependencyParser
from kwja.models.models.discourse_parser import DiscourseParser
from kwja.utils.constants import IGNORE_INDEX


class RelationAnalyzer(nn.Module):
    def __init__(self, hparams: DictConfig, pretrained_model_config: PretrainedConfig) -> None:
        super().__init__()
        self.hparams = hparams

        self.dependency_parser = DependencyParser(
            pretrained_model_config=pretrained_model_config,
            k=hparams.k,
        )
        self.cohesion_analyzer = CohesionAnalyzer(
            pretrained_model_config=pretrained_model_config,
            num_rels=int("pas_analysis" in hparams.cohesion_tasks) * len(hparams.pas_cases)
            + int("coreference" in hparams.cohesion_tasks)
            + int("bridging" in hparams.cohesion_tasks),
        )
        self.discourse_parser = DiscourseParser(pretrained_model_config=pretrained_model_config)

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
                "cohesion_logits": cohesion_logits + (~batch["cohesion_mask"]).float() * -1024.0,
            }
        )
        if "cohesion_target" in batch and "cohesion_mask" in batch:
            cohesion_loss = cohesion_cross_entropy_loss(
                cohesion_logits,
                batch["cohesion_target"],
                batch["cohesion_mask"],
            )
            output.update({"cohesion_loss": cohesion_loss})

        discourse_parsing_logits = self.discourse_parser(pooled_outputs)  # (b, seq, seq, rel)
        output.update(
            {
                "discourse_parsing_logits": discourse_parsing_logits,
            }
        )
        if "discourse_relations" in batch:
            num_labels = (batch["discourse_relations"] != IGNORE_INDEX).sum().item()
            if num_labels:
                discourse_parsing_loss = F.cross_entropy(
                    input=discourse_parsing_logits.view(-1, discourse_parsing_logits.size(3)),
                    target=batch["discourse_relations"].view(-1),
                    ignore_index=IGNORE_INDEX,
                )
            else:
                discourse_parsing_loss = torch.tensor(0.0)
            output.update({"discourse_parsing_loss": discourse_parsing_loss})
        return output
