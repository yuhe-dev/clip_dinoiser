import json
from typing import Dict


class SchemaResolver:
    def load_unified_schema(self, schema_path: str) -> Dict[str, object]:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_dimension_schema(self, schema: Dict[str, object], dimension_name: str) -> Dict[str, object]:
        return dict(schema["dimensions"][dimension_name])
