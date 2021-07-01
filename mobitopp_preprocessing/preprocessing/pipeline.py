import json
from mpipe import UnorderedWorker, UnorderedStage, Pipeline
from esy.osmfilter import run_filter
from esy.osmfilter import Node, Way, Relation


class PbfPoiFilter(UnorderedWorker):
    def __init__(self):
        pass

    def doTask(
            self,
            input_pbf_filepath,
            input_tags_filepath,
            output_json_filepath,
            final_output_filepath
    ):
        filter_tags = self._getPoiTaglist(input_tags_filepath)
        prefilter, blackfilter, whitefilter = self._create_esm_filters(
            filter_tags)
        poi_data = self._extract_poi(
            pbf_filepath=input_pbf_filepath,
            json_filepath=output_json_filepath,
            prefilter=prefilter,
            blackfilter=blackfilter,
            whitefilter=whitefilter
        )
        return poi_data, output_json_filepath, final_output_filepath

    def _get_poi_filter_tags(self, taglist_path):
        with open(taglist_path) as file:
            return json.load(open(file))

    def _create_esm_filters(self, filter_tags):
        prefilter = {
            Node: filter_tags,
            Way: filter_tags,
            Relation: filter_tags
        }
        blackfilter = [()]
        whitefilter = [[()]]

        return prefilter, blackfilter, whitefilter

    def _extract_poi(
        self,
        pbf_filepath,
        json_filepath,
        prefilter,
        blackfilter,
        whitefilter
    ):
        [data, elements] = run_filter(
            elementname="poi",  # TODO: give identifiable name
            PBF_inputfile=pbf_filepath,
            JSON_outputfile=json_filepath,
            prefilter=prefilter,
            blackfilter=blackfilter,
            whitefilter=whitefilter,
            NewPreFilterData=True,
            CreateElements=False,
            LoadElements=False,
            verbose=True
        )
        return data


class DataCleanser(UnorderedWorker):
    def doTask(self, task):
        # TODO implement
        pass


class PreprocessingPipeline:
    _first_stage = None
    _last_stage = None
    _pipeline = None

    def __init__(self):
        pass

    def linkStage(self, stage):
        if self._first_stage is None:
            self._first_stage, self._last_stage = stage, stage
        else:
            self._last_stage.link(stage)
            self._last_stage = stage
        self._pipeline = None

    def run(self, task):
        # TODO: Does it still work when the first stage needs more than
        # 1 parameter
        if self._pipeline is None:
            self._pipeline = Pipeline(self._first_stage)

        self._pipeline.put(task)


def createStage(worker, num_workers):
    return UnorderedStage(worker, num_workers)


def createPipeline(stage):
    return Pipeline(stage)


if __name__ == "__main__":
    pass
