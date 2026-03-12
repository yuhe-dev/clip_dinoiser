from typing import Callable, Dict

import numpy as np

from ..bundle.processed_bundle import ProcessedFeatureBundle
from ..bundle.stats import build_processed_feature_summary
from .encoders import DistributionFeatureEncoder, ProfileFeatureEncoder


class FeaturePostprocessor:
    def process_bundle(
        self,
        raw_bundle,
        dimension_schema: Dict[str, object],
        progress_interval: int = 100,
        log_fn: Callable[[str], None] = print,
    ) -> ProcessedFeatureBundle:
        feature_specs = dict(dimension_schema.get("features", {}))
        label = raw_bundle.dimension_name
        total_records = int(len(raw_bundle.records))
        log_fn("[postprocess] %s: preparing %d raw records" % (label, total_records))

        encoders: Dict[str, object] = {}
        for feature_name, feature_spec in feature_specs.items():
            spec = dict(feature_spec)
            if spec["encoding"] == "distribution":
                log_fn("[postprocess] %s: fitting bin edges for %s" % (label, feature_name))
                encoder = DistributionFeatureEncoder(spec)
                raw_key = str(spec["raw_key"])
                arrays = [
                    np.asarray(record.get(raw_key, np.asarray([], dtype=np.float32)), dtype=np.float32)
                    for record in raw_bundle.records
                ]
                encoder.fit(arrays)
            elif spec["encoding"] == "profile":
                encoder = ProfileFeatureEncoder(spec)
            else:
                raise ValueError("Unsupported encoding='%s'" % spec["encoding"])
            encoders[feature_name] = encoder

        processed_records = []
        for idx, record in enumerate(raw_bundle.records, start=1):
            feature_blocks = {}
            for feature_name, feature_spec in feature_specs.items():
                spec = dict(feature_spec)
                raw = np.asarray(record.get(spec["raw_key"], np.asarray([], dtype=np.float32)), dtype=np.float32)
                feature_blocks[feature_name] = encoders[feature_name].transform(raw, record)
            processed_records.append(
                {
                    "image_rel": record.get("image_rel", ""),
                    "annotation_rel": record.get("annotation_rel", ""),
                    "schema_version": str(dimension_schema["schema_version"]),
                    "features": feature_blocks,
                }
            )
            if progress_interval > 0 and (idx == total_records or idx % progress_interval == 0):
                log_fn("[postprocess] %s: processed %d/%d records" % (label, idx, total_records))

        return ProcessedFeatureBundle(
            dimension_name=raw_bundle.dimension_name,
            records=processed_records,
            schema=dimension_schema,
            processing_config={
                "dimension": raw_bundle.dimension_name,
                "schema_version": dimension_schema["schema_version"],
            },
            summary=build_processed_feature_summary(processed_records),
        )
