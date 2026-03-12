from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RawFeatureBundle:
    dimension_name: str
    records: List[Dict[str, object]]
    stats: Dict[str, object]
    feature_config: Dict[str, object]
