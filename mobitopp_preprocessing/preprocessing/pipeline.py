import json
from mpipe import UnorderedWorker, UnorderedStage, Pipeline
from esy.osmfilter import run_filter
from esy.osmfilter.osm_filter import create_single_element
from esy.osmfilter import Node, Way, Relation


from mobitopp_preprocessing.helpers.file_helper import (
    file_exists,
    read_json_file,
    save_as_json_file,
)


class Filter(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, task):
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


class NotJsonFileError(Exception):
        pass


class NoStageDefinedError(Exception):
    """Raised when no stage has been defined for the pipeline."""

        pass


class Pipeline:
    def __init__(self, stages=[]):
        self._stages = stages

    def add(self, stage):
        self.stages.append(stage)

    def run(self, task):
        if len(self._stages) == 0:
            raise NoStageDefinedError("No stage has been defined on this pipeline!")

        return self._execute(task=task, filters=self._stages)

    def _execute(self, task, filters):
        filter, *tail = filters
        result = filter.execute(task)

        if len(tail) != 0:
            return self._execute(result, tail)
        else:
            return result

        self._pipeline.put(task)


def createStage(worker, num_workers):
    return UnorderedStage(worker, num_workers)


def createPipeline(stage):
    return Pipeline(stage)


if __name__ == "__main__":
    pass
