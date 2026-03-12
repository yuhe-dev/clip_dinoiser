from ..bundle import (
    ProcessedBundleIO,
    RawBundleIO,
    RawFeatureBundle,
    build_raw_feature_stats,
)
from .config import PipelineConfig
from .factory import FeaturePipelineFactory


class DataFeaturePipelineRunner:
    def __init__(self, factory=None):
        self.factory = factory or FeaturePipelineFactory()

    def run_raw(
        self,
        dimension_name,
        subset_root,
        subset_records,
        data_root,
        index_path,
        feature_meta,
        progress_interval=100,
        show_progress=True,
    ):
        pipeline_config = PipelineConfig(
            subset_root=subset_root,
            index_path=index_path,
            data_root=data_root,
            feature_meta=dict(feature_meta),
            progress_interval=int(progress_interval),
        )
        extractor = self.factory.create_raw_extractor(dimension_name, pipeline_config)
        records = extractor.extract_records(
            subset_root=subset_root,
            subset_records=subset_records,
            feature_meta=feature_meta,
            show_progress=show_progress,
            progress_interval=progress_interval,
        )
        feature_keys = tuple(
            key
            for key in records[0].keys()
            if key.endswith("_raw") or key.endswith("_num_values")
        ) if records else tuple()
        stats_feature_keys = tuple(key for key in feature_keys if key.endswith("_raw"))
        bundle = RawFeatureBundle(
            dimension_name=dimension_name,
            records=records,
            stats=build_raw_feature_stats(records, stats_feature_keys) if stats_feature_keys else {"num_samples": len(records), "features": {}},
            feature_config={
                "subset_root": subset_root,
                "index_path": index_path,
                "feature_meta": dict(feature_meta),
                "records_file": "%s_raw_features.npy" % dimension_name,
                "stats_file": "%s_global_stats.json" % dimension_name,
            },
        )
        output_root = "%s/%s" % (data_root, dimension_name)
        return RawBundleIO().save(bundle, output_root)

    def run_postprocess(
        self,
        dimension_name,
        data_root,
        schema_path,
        progress_interval=100,
    ):
        raw_root = "%s/%s" % (data_root, dimension_name)
        raw_records_path = "%s/%s_raw_features.npy" % (raw_root, dimension_name)
        raw_stats_path = "%s/%s_global_stats.json" % (raw_root, dimension_name)
        raw_config_path = "%s/%s_feature_config.json" % (raw_root, dimension_name)
        raw_bundle = RawBundleIO().load(dimension_name, raw_root)
        dimension_schema = self.factory.load_dimension_schema(schema_path, dimension_name)
        postprocessor = self.factory.create_postprocessor(schema_path)
        bundle = postprocessor.process_bundle(raw_bundle, dimension_schema, progress_interval=progress_interval)
        bundle.processing_config.update(
            {
                "source_records_path": raw_records_path,
                "source_stats_path": raw_stats_path,
                "source_config_path": raw_config_path,
                "schema_source_path": schema_path,
            }
        )
        return ProcessedBundleIO().save(bundle, raw_root)

    def run_full(
        self,
        dimension_name,
        subset_root,
        subset_records,
        data_root,
        index_path,
        feature_meta,
        schema_path,
        progress_interval=100,
        show_progress=True,
    ):
        raw_result = self.run_raw(
            dimension_name=dimension_name,
            subset_root=subset_root,
            subset_records=subset_records,
            data_root=data_root,
            index_path=index_path,
            feature_meta=feature_meta,
            progress_interval=progress_interval,
            show_progress=show_progress,
        )
        processed_result = self.run_postprocess(
            dimension_name=dimension_name,
            data_root=data_root,
            schema_path=schema_path,
            progress_interval=progress_interval,
        )
        return {
            "raw": raw_result,
            "processed": processed_result,
        }
