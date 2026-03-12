import json
import os
from typing import Dict

import numpy as np

from .processed_bundle import ProcessedFeatureBundle
from .raw_bundle import RawFeatureBundle


class RawBundleIO:
    def save(self, bundle: RawFeatureBundle, output_root: str) -> Dict[str, str]:
        os.makedirs(output_root, exist_ok=True)
        dim = bundle.dimension_name
        records_path = os.path.join(output_root, f"{dim}_raw_features.npy")
        stats_path = os.path.join(output_root, f"{dim}_global_stats.json")
        config_path = os.path.join(output_root, f"{dim}_feature_config.json")

        np.save(records_path, np.asarray(list(bundle.records), dtype=object), allow_pickle=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(bundle.stats, f, indent=2, ensure_ascii=False)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(bundle.feature_config, f, indent=2, ensure_ascii=False)
        return {
            "records_path": records_path,
            "stats_path": stats_path,
            "config_path": config_path,
        }

    def load(self, dimension_name: str, input_root: str) -> RawFeatureBundle:
        records_path = os.path.join(input_root, "%s_raw_features.npy" % dimension_name)
        stats_path = os.path.join(input_root, "%s_global_stats.json" % dimension_name)
        config_path = os.path.join(input_root, "%s_feature_config.json" % dimension_name)
        records = np.load(records_path, allow_pickle=True)
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            feature_config = json.load(f)
        return RawFeatureBundle(
            dimension_name=dimension_name,
            records=[dict(item) for item in records.tolist()],
            stats=stats,
            feature_config=feature_config,
        )


class ProcessedBundleIO:
    def save(self, bundle: ProcessedFeatureBundle, output_root: str) -> Dict[str, str]:
        os.makedirs(output_root, exist_ok=True)
        dim = bundle.dimension_name
        records_path = os.path.join(output_root, f"{dim}_processed_features.npy")
        schema_path = os.path.join(output_root, f"{dim}_processed_schema.json")
        config_path = os.path.join(output_root, f"{dim}_processing_config.json")
        summary_path = os.path.join(output_root, f"{dim}_processed_summary.json")

        np.save(records_path, np.asarray(list(bundle.records), dtype=object), allow_pickle=True)
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(bundle.schema, f, indent=2, ensure_ascii=False)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(bundle.processing_config, f, indent=2, ensure_ascii=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(bundle.summary, f, indent=2, ensure_ascii=False)
        return {
            "records_path": records_path,
            "schema_path": schema_path,
            "config_path": config_path,
            "summary_path": summary_path,
        }

    def load(self, dimension_name: str, input_root: str) -> ProcessedFeatureBundle:
        records_path = os.path.join(input_root, "%s_processed_features.npy" % dimension_name)
        schema_path = os.path.join(input_root, "%s_processed_schema.json" % dimension_name)
        config_path = os.path.join(input_root, "%s_processing_config.json" % dimension_name)
        summary_path = os.path.join(input_root, "%s_processed_summary.json" % dimension_name)
        records = np.load(records_path, allow_pickle=True)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            processing_config = json.load(f)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return ProcessedFeatureBundle(
            dimension_name=dimension_name,
            records=[dict(item) for item in records.tolist()],
            schema=schema,
            processing_config=processing_config,
            summary=summary,
        )
