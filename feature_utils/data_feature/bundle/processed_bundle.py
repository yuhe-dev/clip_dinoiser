from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProcessedFeatureBundle:
    dimension_name: str
    records: List[Dict[str, object]]
    schema: Dict[str, object]
    processing_config: Dict[str, object]
    summary: Dict[str, object]
