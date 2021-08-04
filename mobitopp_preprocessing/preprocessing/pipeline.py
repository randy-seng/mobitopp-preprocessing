import os
import logging
from pathlib import Path

import pandas as pd
import geopandas as gpd
from abc import ABCMeta, abstractmethod
from esy.osmfilter import run_filter
from esy.osmfilter.osm_filter import create_single_element
from esy.osmfilter import Node, Way, Relation
from pyrosm import OSM
from pygeos import Geometry


from mobitopp_preprocessing.helpers.file_helper import (
    file_exists,
    read_json_file,
    save_as_json_file,
)


class Filter(metaclass=ABCMeta):
    @abstractmethod
    def execute(self, task):
        pass


class PbfFilter(Filter):
    """Reads from a osm pbf file and filters out POIs using a tag list.

    This class should be used in conjunction with the createStage function,
    returning a pipeline stage which can then be used to add it to the
    existing class Pipeline.

    Arguments:
    input_pbf_filepath,
    input_poi_tags_filepath,
    output_json_filepath,
    final_output_filepath
    """

    def __init__(
        self,
        output_dir,
        output_file_name,
        prefilter_tags_path,
        whitefilter_tags_path=None,  # TODO: Change to None
        blackfilter_tags_path=None,  # TODO: Change to None
    ):
        # TODO: setup
        self._prefilter_tags_path = prefilter_tags_path
        self._output_dir = output_dir
        self._output_file_name = output_file_name
        self._whitefilter_tags_path = whitefilter_tags_path
        self._blackfilter_tags_path = blackfilter_tags_path

    def execute(self, input_pbf_filepath):
        json_filepath = os.path.join(self._output_dir, self._output_file_name + ".json")
        prefilter, blackfilter, whitefilter = self._create_esm_filters()
        poi_data = self._filter_data(
            pbf_filepath=input_pbf_filepath,
            json_filepath=json_filepath,
            prefilter=prefilter,
            blackfilter=blackfilter,
            whitefilter=whitefilter,
        )
        return [poi_data, json_filepath]

    def _create_esm_filters(self):
        prefilter = self._create_prefilter(self._prefilter_tags_path)
        whitefilter = self._create_whitefilter(self._whitefilter_tags_path)
        blackfilter = self._create_blackfilter(self._blackfilter_tags_path)

        return prefilter, blackfilter, whitefilter

    def _create_prefilter(self, prefilter_tags_path):
        prefilter_tags = read_json_file(prefilter_tags_path)
        return {Node: prefilter_tags, Way: prefilter_tags, Relation: prefilter_tags}

    def _create_whitefilter(self, whitefilter_tags_path):
        if whitefilter_tags_path is None:
            return [[()]]
        else:
            whitefilter_tags = read_json_file(self._whitefilter_tags_path)
            # TODO: Transform whitefilter tags to adhere to esy-osmfilter whitefilter

    def _create_blackfilter(self, blackfilter_tags_path):
        if blackfilter_tags_path is None:
            return [()]
        else:
            blackfilter_tags = read_json_file(self._blackfilter_tags_path)
            # TODO: Transform blackfilter tags to adhere to esy-osmfilter blackfilter

    def _filter_data(
        self, pbf_filepath, json_filepath, prefilter, blackfilter, whitefilter
    ):

        [data, elements] = run_filter(
            elementname=self._output_file_name.join(
                "_poi"
            ),  # TODO: give identifiable name
            PBF_inputfile=pbf_filepath,
            JSON_outputfile=json_filepath,
            prefilter=prefilter,
            blackfilter=blackfilter,
            whitefilter=whitefilter,
            NewPreFilterData=True,
            CreateElements=False,
            LoadElements=False,
            verbose=True,
        )
        return data, elements


class CalculateAttractivity(Filter):
    def __init__(
        self,
        poi_processing_info_csv,
        output_dir,
        output_file_name,
        pbf_path,
        poi_filter_tags_path,
        poi_path=None,
        use_cached_data=False,
        epsg=3035,
    ):
        self._poi_path = poi_path
        self._poi_processing_info = poi_processing_info_csv
        self._output_dir = output_dir
        self._output_file_name = output_file_name
        self._poi_filter_tags = poi_filter_tags_path
        self._use_cached_data = use_cached_data
        self._epsg = epsg
        self._building_data = self.load_building_data(pbf_path)
        self._poi_data = self.load_poi_data(pbf_path, poi_filter_tags_path)

    def load_building_data(self, pbf_path):
        buildings_path = os.path.join(self._output_dir, "buildings.geojson")
        selected_col = ["id", "geometry"]

        if self._use_cached_data and file_exists(buildings_path):
            buildings = gpd.read_file(buildings_path)
            buildings.set_crs(epsg=self._epsg, inplace=True)
            buildings.set_index("id", inplace=True)
            return gpd.read_file(buildings_path)
        else:
            # load osm building data from scratch
            osm = OSM(pbf_path)
            buildings = osm.get_buildings()
            buildings = buildings[selected_col]
            buildings.to_crs(epsg=self._epsg, inplace=True)
            buildings.set_index("id", inplace=True)
            buildings.to_file(buildings_path, driver="GeoJSON")
            return buildings

    def load_poi_data(self, pbf_path, tag_list_path):
        pois_path = os.path.join(self._output_dir, "pois.csv")
        selected_col = ["id", "geometry"]

        if self._use_cached_data and file_exists(pois_path):
            df = pd.read_csv(pois_path)
            df["geometry"] = df["geometry"].apply(Geometry)
            pois = gpd.GeoDataFrame(df, crs="epsg:{}".format(self._epsg))
            pois.set_index("id", inplace=True)
            return pois
        else:
            # load osm poi data from scratch
            taglist = read_json_file(tag_list_path)
            osm = OSM(pbf_path)
            # pois = osm.get_pois(custom_filter=taglist)
            pois = osm.get_data_by_custom_criteria(custom_filter=taglist)
            pois = pois[selected_col]
            pois.to_crs(epsg=self._epsg, inplace=True)
            pois.set_index("id", inplace=True)
            pois.to_csv(pois_path)
            return pois

    def execute(self, poi=None):
        poi_list = []

        if poi is None:
            poi = read_json_file(self._poi_path)

        processing_info = pd.read_csv(self._poi_processing_info)

        for row in processing_info.itertuples():
            result = self._process(attractivity_info=row, prefiltered_data=poi)
            poi_list.append(result)

        save_as_json_file(
            self._output_dir,
            self._output_file_name + "_with_attractivity.json",
            poi_list,
        )
        print("finished")

    def _process(self, attractivity_info, prefiltered_data):
        filters = read_json_file(os.path.join(os.getcwd(), attractivity_info.filters))
        whitefilter = self._create_whitefilter(filters["whitefilter"])
        blackfilter = self._create_blackfilter(filters.get("blackfilter"))
        poi_desc = attractivity_info.name

        filtered = create_single_element(
            data=prefiltered_data,
            JSON_outputfile=os.path.join(self._output_dir, "filter_results/"),
            elementname=self._output_file_name + "_{}".format(poi_desc),
            whitefilter=whitefilter,
            blackfilter=blackfilter,
        )
        pois_with_attractivity = self._calculate_attractivity(
            pois=filtered,
            calculation_type=attractivity_info.calculation_type,
            weighting_factor=attractivity_info.weighting_factor,
        )
        return pois_with_attractivity

    def _create_whitefilter(self, whitefilter_data):
        whitefilter = []
        for key, tags in whitefilter_data.items():
            whitefilter_elem = map(lambda tag: [(key, tag)], tags)
            whitefilter.extend(whitefilter_elem)
        return whitefilter

    def _create_blackfilter(self, blackfilter_data):
        if blackfilter_data is None:
            return []

        blackfilter = []

        for key, tags in blackfilter_data.items():
            blackfilter_elem = map(lambda tag: (key, tag), tags)
            blackfilter.extend(blackfilter_elem)
        return blackfilter

    def _calculate_attractivity(self, pois, calculation_type, weighting_factor):
        # TODO: use weighting factor of activity_types.csv if no weighting factor
        # specified

        logger = logging.getLogger("calculate_attractivity_logger")
        logger.setLevel(logging.DEBUG)

        if calculation_type == "num_pois":
            pois_count = len(pois["Node"]) + len(pois["Way"]) + len(pois["Relation"])
            attractivity = pois_count * weighting_factor

            # Nodes
            for id_node in pois["Node"].keys():
                pois["Node"][id_node]["attractivity"] = attractivity

            # Ways
            for id_node in pois["Way"].keys():
                pois["Node"][id_node]["attractivity"] = attractivity

            # Relations
            for id_node in pois["Relation"].keys():
                pois["Node"][id_node]["attractivity"] = attractivity

            return pois

        elif calculation_type == "area":
            # TODO: Implement attractivity for area
            """for id_node in pois["Node"].keys():
            return pois"""

            # Nodes
            node_ids = list(map(int, pois["Node"].keys()))
            try:
                gd_pois = self._poi_data.loc[node_ids]
                joined = gpd.sjoin(
                    self._building_data, gd_pois, how="inner", op="contains"
                )
                joined.set_index("index_right", inplace=True)
                # geometry.area unit dependant on epsg
                joined["area"] = joined.geometry.area
            except KeyError as err:
                logger.error("Check for missing tags in taglist", err)
                return pois

            for id in joined.index:
                poi_area = joined["area"].loc[id]
                poi_id = str(id)  # must be string in pois dict

                pois["Node"][poi_id]["area"] = poi_area
                pois["Node"][poi_id]["attractivity"] = weighting_factor * poi_area

            # Ways
            way_ids = list(map(int, pois["Way"].keys()))
            gd_ways = self._poi_data.loc[way_ids]
            gd_ways["area"] = gd_ways.geometry.area
            ic(gd_ways.head(30))
            ic(gd_ways.index)
            ic(type(gd_ways.index))

            for id in gd_ways.index:
                poi_id = str(id)
                poi_area = gd_ways["area"].loc[id]

                pois["Way"][poi_id]["area"] = poi_area
                pois["Way"][poi_id]["attractivity"] = weighting_factor * poi_area

            # Relations
            relation_ids = list(map(int, pois["Relation"].keys()))
            gd_relations = self._poi_data.loc[relation_ids]
            gd_relations["area"] = gd_relations.geometry.area
            ic(gd_relations.head(30))
            ic(gd_relations.index)

            for id in gd_relations.index:
                poi_id = str(id)
                poi_area = gd_ways["area"].loc[id]

                pois["Relation"][poi_id]["area"] = poi_area
                pois["Relation"][poi_id]["attractivity"] = weighting_factor * poi_area

            return pois


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


def main():
    input_pbf_file = os.path.join(
        os.getcwd(), "data/osm/pbf_files/liechtenstein-140101.osm.pbf"
    )
    json_taglist = Path(__file__).parents[0] / "resources" / "poi_filter_tags.json"
    output_dir = os.path.join(os.getcwd(), "data/osm/liechtenstein")
    poi_attractivity_info = os.path.join(
        os.getcwd(),
        "mobitopp_preprocessing/preprocessing/resources/poi_attractivity_info.csv",
    )
    liechtenstein_poi_path = "/Users/jibi/dev_projects/mobitopp/mobitopp-preprocessing/data/osm/liechtenstein/liechtenstein_poi.json"

    poi_processing_pipeline = Pipeline(
        [PbfFilter(output_dir, "liechtenstein_poi", json_taglist)]
    )
    # poi_processing_pipeline.run(input_pbf_file)

    """ attractivity_calculator = CalculateAttractivity(
        poi_processing_info_csv=poi_attractivity_info,
        output_dir=output_dir,
        output_file_name="liechtenstein_filtered_poi",
        pbf_path=input_pbf_file,
        poi_filter_tags_path=json_taglist,
        poi_path=liechtenstein_poi_path,
        use_cached_data=True,
    )

    attractivity_calculator.execute() """

    karlsruhe = "/Users/jibi/dev_projects/mobitopp/mobitopp-preprocessing/data/osm/pbf_files/karlsruhe-regbez-210614.osm.pbf"
    out_ka = os.path.join(os.getcwd(), "data/osm/karlsruhe")

    ka_poi_filter = PbfFilter(out_ka, "karlsruhe_poi", json_taglist)
    ka_attractivity_filter = CalculateAttractivity(
        poi_attractivity_info, out_ka, "ka_attractivity", karlsruhe, json_taglist
    )

    ka_poi_pipeline = [ka_poi_filter, ka_attractivity_filter]
    ka_poi_pipeline.run(karlsruhe)


if __name__ == "__main__":
    main()
