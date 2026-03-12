from ..extraction import CoverageRawExtractor, DifficultyRawExtractor, QualityRawExtractor
from ..postprocess import FeaturePostprocessor, SchemaResolver


class FeaturePipelineFactory:
    def create_raw_extractor(self, dimension_name, pipeline_config):
        del pipeline_config
        if dimension_name == "quality":
            return QualityRawExtractor()
        if dimension_name == "difficulty":
            return DifficultyRawExtractor()
        if dimension_name == "coverage":
            return CoverageRawExtractor()
        raise ValueError("Unsupported dimension '%s'" % dimension_name)

    def create_postprocessor(self, schema_path):
        del schema_path
        return FeaturePostprocessor()

    def load_dimension_schema(self, schema_path, dimension_name):
        schema = SchemaResolver().load_unified_schema(schema_path)
        return SchemaResolver().get_dimension_schema(schema, dimension_name)
