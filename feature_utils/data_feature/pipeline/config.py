from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PipelineConfig:
    subset_root: Optional[str] = None
    index_path: Optional[str] = None
    data_root: str = "./data/data_feature"
    schema_path: str = "./docs/feature_schema/unified_processed_feature_schema.json"
    feature_meta: Dict[str, object] = field(default_factory=dict)
    progress_interval: int = 100
