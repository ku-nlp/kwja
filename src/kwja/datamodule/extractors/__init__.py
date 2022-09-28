from typing import Union

from .bridging import BridgingAnnotation, BridgingExtractor
from .coreference import CoreferenceAnnotation, CoreferenceExtractor
from .pas import PasAnnotation, PasExtractor

Annotation = Union[PasAnnotation, BridgingAnnotation, CoreferenceAnnotation]

__all__ = [
    "PasExtractor",
    "PasAnnotation",
    "BridgingExtractor",
    "BridgingAnnotation",
    "CoreferenceExtractor",
    "CoreferenceAnnotation",
    "Annotation",
]
