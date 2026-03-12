from typing import Callable, Dict, List, Sequence


class BaseRawExtractor:
    dimension_name = ""

    def __init__(self, feature_factory: Callable[[dict], Dict[str, object]]):
        self.feature_factory = feature_factory

    def build_feature_instances(self, feature_meta: dict) -> Dict[str, object]:
        return dict(self.feature_factory(feature_meta))

    def load_sample_context(self, subset_root: str, record: dict) -> dict:
        raise NotImplementedError

    def extract_single_record(
        self,
        record: dict,
        sample_context: dict,
        feature_instances: Dict[str, object],
        feature_meta: dict,
    ) -> dict:
        raise NotImplementedError

    def extract_records(
        self,
        subset_root: str,
        subset_records: Sequence[dict],
        feature_meta: dict,
        show_progress: bool = True,
        progress_interval: int = 100,
    ) -> List[dict]:
        del show_progress
        del progress_interval
        feature_instances = self.build_feature_instances(feature_meta)
        extracted: List[dict] = []
        for record in subset_records:
            sample_context = self.load_sample_context(subset_root, record)
            if sample_context is None:
                continue
            extracted.append(
                self.extract_single_record(
                    record=record,
                    sample_context=sample_context,
                    feature_instances=feature_instances,
                    feature_meta=feature_meta,
                )
            )
        return extracted
