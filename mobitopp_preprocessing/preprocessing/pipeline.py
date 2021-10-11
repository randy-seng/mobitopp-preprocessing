import os
import logging
from pathlib import Path
from typing import Union

from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from abc import ABCMeta, abstractmethod
from esy.osmfilter import run_filter
from esy.osmfilter.osm_filter import create_single_element
from esy.osmfilter import Node, Way, Relation
from pyrosm import OSM
import shapely
import srtm
import ijson
from more_itertools import chunked


from mobitopp_preprocessing.helpers.file_helper import (
    createDir,
    createFile,
    file_exists,
    read_json_file,
    save_as_json_file,
    save_gdf_to_geojson,
    save_gdf_to_csv,
)

from mobitopp_preprocessing.preprocessing.config import URBAN_ROAD_NETWORK_DEFAULT


class NotJsonFileError(Exception):
    pass


class NoStageDefinedError(Exception):
    """Raised when no stage has been defined for the pipeline."""

    pass


class Filter(metaclass=ABCMeta):
    """
    An abstract class as in the Pipes and Filters architectural pattern.
    Concrete classes implementing this class receive input data and filter,
    transform, enrich or enhance said data.

    The concrete classes implementing this abstract class can be found in this
    Python module.

    Furthermore, this class should be used in conjunction with the Pipeline
    class.

    When subclassing this class, it is needed to implement the following two
    abstract methods:
        - the filter method
        - _save method

    Args:
        write_result (bool): A flag used whether to write the processed data to disk.
        out_dir (str): Specifies the output directory where the data should be saved
        to.

    ### Example:
    Assume having a class called ExampleFilter subclassing the abstract Filter
    class::
        data = [1, 2, 3, 4]

        filter1 = ExampleFilter(write_result=True, out_dir="path/to/out")

        # calls the filter method on data and saves the processed data to disk
        filter1(data)

    """

    def __init__(self, write_result: bool = False, out_dir: str = None) -> None:
        self._write_result = write_result
        self._out_dir = out_dir

        if out_dir is not None:
            createDir(out_dir)

    def __call__(self, data):
        """
        Applies the filter method to the data and returns the result.

        NOTE: Implementing the __call__ method allows instances of this class
        to behave like functions.

        Acts as a template method and calls the abstract method filter.

        ### Example::

            # Class ExampleFilter inherits __call__ method from super class Filter
            # This makes object instances of this class able to be treated as function

            class CalculateSum(Filter):
                def filter(data):
                    return sum(data)


            sum_filter = CalculateSum()
            data = [1, 2, 3, 4]

            #  executes the __call__ method when using object instance as method
            sum_filter(data)


        """
        for datum in data:
            result = self.filter(datum)
            self.save(result)
            yield result

    @abstractmethod
    def filter(self, data):
        """
        Depending on the implementation, this abstract method enhances, filters,
        or transforms the data and returns data.

        Args:
            data: The data to undergo manipulation

        Returns:
            the data after manipulation(enhancement, filtering, or transformation)

        """
        pass

    def save(self, data):
        """
        Saves the data after applying the filter method at out_dir if
        write_result is True.

        Acts as a template method and calls the abstract method _save.

        """
        if self._write_result:
            self._save(data)

    @abstractmethod
    def _save(self, data) -> None:
        """
        An abstract method where concrete implementations should
        save the data to disk.

        """
        pass


class DataSource(metaclass=ABCMeta):
    """
    An abstract class as in the Pipes and Filters architectural pattern.

    This class acts as a data source and delivers input upon request to
    the data processing pipeline.

    The concrete classes implementing this abstract class can be found in this
    Python module.

    Furthermore, this class is used by the Pipeline class.

    Subclasses extending this class need to implement the abstract method read().

    Args:
            data_source: the data source to be read from.

    """

    def __init__(self, data_source) -> None:
        super().__init__()
        self._data_source = data_source

    @abstractmethod
    def read(self):
        """
        Reads data from the data source and returns it.

        An abstract method to be implemented by the concrete classes.
        """
        pass


class DataSink(metaclass=ABCMeta):
    """
    An abstract class as in the Pipes and Filters architectural pattern.

    This class acts as a data sink and consumes the data that it receives.
    Depending on the implementation consuming data could be writing it to
    a file and saving it or outputting the data to the console.

    The concrete classes implementing this abstract class can be found in this
    Python module.

    Furthermore, this class is used by the Pipeline class.

    Subclasses extending this class need to implement the abstract method consume().

    """

    def __call__(self, data) -> None:
        """
        Consumes the data.

        This method enables an instance of this class to be treated as a
        method.

        Example::

            data = [1, 2, 3, 4]
            data_sink = ExampleDataSink() # concrete implementation of DataSink

            data_sink(data) # consumes data
        """
        self._consume(data)

    @abstractmethod
    def _consume(self, data):
        """
        Consumes the data.

        It is an abstract class.

        """
        pass


class FileDataSource(DataSource):
    """
    Acts as a wrapper for a file path.

    Args:
        data_source (str): The file path

    """

    def __init__(self, data_source: str) -> None:
        if not file_exists(data_source):
            raise FileNotFoundError(filename=str)

        super().__init__(data_source)

    def read(self) -> str:
        """
        Returns the file path.
        """
        return [self._data_source]


class GeoDataFrameStream(DataSource):
    """
    This class creates a stream of GeoDataFrames by reading from a GeoJSON
    file and feeding it as an input to the pipeline.


    Args:
        geojson_path (str): The path to the geojson file.
        chunk_size (int): The chunk size which the stream returns per iteration.

    """

    def __init__(self, geojson_path: str, chunk_size: int = 100) -> None:
        super().__init__(geojson_path)
        self._chunk_size = chunk_size

    def read(self):
        """Yields GeodataFrames as generator by reading from geojson_path."""
        for gdf in self._geojson_generator(self._data_source, self._chunk_size):
            print(gdf.head())
            yield gdf

    def _geojson_generator(self, geojson_path: str, chunk_size):
        """
        Yields GeoDataFrames of size=chunk_size by reading from geojson
        file.
        """
        if chunk_size == 0:
            yield gpd.read_file(geojson_path)
        else:
            geojson_type = self._get_type(geojson_path)
            geojson_crs = self._get_crs(geojson_path)

            for feature_chunk in self._geo_feature_gen(geojson_path, chunk_size):
                geojson_chunk = {
                    "type": geojson_type,
                    "crs": geojson_crs,
                    "features": feature_chunk,
                }
                yield gpd.GeoDataFrame.from_features(geojson_chunk)

    def _geo_feature_gen(self, geojson_path, chunk_size):
        """
        Extract 'features' of geojson file and yield json objects of size chunk_size.
        """
        with open(geojson_path, "rb") as in_file:
            feature_generator = ijson.items(in_file, "features.item")
            for chunk in chunked(feature_generator, chunk_size):
                yield chunk

    def _get_type(self, geojson_path):
        """
        Returns the value of key 'type' in the geojson file.
        Otherwise return None.
        """
        with open(geojson_path, "rb") as in_file:
            geojson_type = None
            geojson_type_gen = ijson.items(in_file, "type")
            for geo_type in geojson_type_gen:
                geojson_type = geo_type
            in_file.close()
            return geojson_type

    def _get_crs(self, geojson_path):
        """
        Returns the crs information if it exists in the geojson file.
        Otherwise return None.
        """
        with open(geojson_path, "rb") as in_file:
            geojson_crs = None
            geojson_crs_gen = ijson.items(in_file, "crs")
            for geo_crs in geojson_crs_gen:
                geojson_crs = geo_crs
            in_file.close()
            return geojson_crs


class WriteGdfToFile(DataSink):
    """
    Writes a geopandas GeoDataFrame to a file.

    Args:
        out_dir (str): the directory where the file is to be saved.
        file_name (str): the file name.
    """

    def __init__(self, out_dir, file_name) -> None:
        self._out_dir = out_dir
        self._file_name = file_name

    def _consume(self, data) -> None:
        """
        Saves a GeoDataFrame as file to the directory specified in the class attribute
        _out_dir and _file_name.

        Args:
            data (GeoDataFrame): the GeoDataFrame to be saved.
        """
        save_gdf_to_geojson(data, self._out_dir, self._file_name)


class WriteGdfStreamToFile(DataSink):
    """
    Consumes a stream of GeoDataFrames an writes it to one file.

    Args:
        output_file_path (str): The file path where the GeoDataFrame should be saved.
    """

    def __init__(self, output_file_path) -> None:
        self._output_file_path = output_file_path
        createFile(output_file_path)

    def _consume(self, data):
        """
        Receives a stream of GeoDataFrames and writes it to a geojson file.
        """
        with open(self._data_sink, "w") as out_file:
            for gdf in data:
                as_geojson = gdf.to_json()


class PbfPoiFilter(Filter):
    """
    Reads from a OpenStreetMap pbf file and filters out POI data.

    A JSON file containing a list of OSM key value tags is used to filter out

    Args:
        input_pbf_filepath,
        input_poi_tags_filepath,
        output_json_filepath,
        final_output_filepath
    """

    def __init__(
        self,
        out_dir,
        prefilter_tags_path,
        out_file_name: str = "PbfPoiFilter",
        whitefilter_tags_path=None,
        blackfilter_tags_path=None,
        write_result=False,
    ):
        super().__init__(write_result=write_result, out_dir=out_dir)
        self._prefilter_tags_path = prefilter_tags_path
        self._out_file_name = out_file_name
        self._whitefilter_tags_path = whitefilter_tags_path
        self._blackfilter_tags_path = blackfilter_tags_path

    def filter(self, input_pbf_filepath):
        json_filepath = os.path.join(self._out_dir, self._out_file_name + ".json")
        prefilter, blackfilter, whitefilter = self._create_esm_filters()
        poi_data = self._filter_data(
            pbf_filepath=input_pbf_filepath,
            json_filepath=json_filepath,
            prefilter=prefilter,
            blackfilter=blackfilter,
            whitefilter=whitefilter,
        )
        return poi_data

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

        [data, _] = run_filter(
            elementname=self._out_file_name.join("_poi"),
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
        return data

    def _save(self, data):
        save_as_json_file(self._out_dir, self._out_file_name, data)


class PbfPoiFilter2(Filter):
    def __init__(
        self,
        out_dir: str,
        poi_filter_tags_path: str,
        out_file_name: str = "PbfPoiFilter2",
        write_result: bool = False,
    ) -> None:
        super().__init__(write_result=write_result, out_dir=out_dir)
        self._poi_filter_tags_path = poi_filter_tags_path
        self._out_file_name = out_file_name

    def filter(self, in_pbf_path):
        osm = OSM(in_pbf_path)
        poi_filter_tags = self._create_prefilter(self._poi_filter_tags_path)
        poi_data = osm.get_data_by_custom_criteria(poi_filter_tags)
        return poi_data

    def _create_prefilter(self, poi_filter_tags_path):
        prefilter_tags = read_json_file(poi_filter_tags_path)
        return {Node: prefilter_tags, Way: prefilter_tags, Relation: prefilter_tags}

    def _save(self, data) -> None:
        save_gdf_to_geojson(data, self._out_dir, self.out_file_name)


class CalculateAttractivity(Filter):
    def __init__(
        self,
        poi_attractivity_info_path: str,
        out_dir: str,
        pbf_path: str,
        poi_filter_tags_path: str,
        out_file_name: str = "CalculateAttractivity",
        epsg: int = 3035,
        write_result: bool = False,
    ):
        super().__init__(write_result=write_result, out_dir=out_dir)
        self._poi_attractivity_info_path = poi_attractivity_info_path
        self._out_file_name = out_file_name + "_with_attractivity"
        self._poi_filter_tags = poi_filter_tags_path
        self._epsg = epsg
        self._building_data = self.load_building_data(pbf_path)
        self._poi_data = self.load_poi_data(pbf_path, poi_filter_tags_path)

    def filter(self, poi):
        poi_list = []

        attractivity_info = pd.read_csv(self._poi_attractivity_info_path)

        for row in tqdm(
            attractivity_info.itertuples(),
            "Calculate attractivities for POIs",
        ):
            result = self._process(attractivity_info=row, prefiltered_data=poi)
            poi_list.append(result)

        return poi_list

    def load_building_data(self, pbf_path):
        selected_col = ["id", "geometry"]

        # load osm building data from scratch
        osm = OSM(pbf_path)
        buildings = osm.get_buildings()
        buildings = buildings[selected_col]
        buildings.to_crs(epsg=self._epsg, inplace=True)
        buildings.set_index("id", inplace=True)
        return buildings

    def load_poi_data(self, pbf_path, taglist_path):
        selected_col = ["id", "geometry"]

        # load osm poi data from scratch
        taglist = read_json_file(taglist_path)
        osm = OSM(pbf_path)
        pois = osm.get_data_by_custom_criteria(custom_filter=taglist)
        pois = pois[selected_col]
        pois.to_crs(epsg=self._epsg, inplace=True)
        pois.set_index("id", inplace=True)
        return pois

    def _process(self, attractivity_info, prefiltered_data):
        filters = read_json_file(os.path.join(os.getcwd(), attractivity_info.filters))
        whitefilter = self._create_whitefilter(filters["whitefilter"])
        blackfilter = self._create_blackfilter(filters.get("blackfilter"))
        poi_desc = attractivity_info.name

        filtered = create_single_element(
            data=prefiltered_data,
            JSON_outputfile=os.path.join(self._out_dir, "filter_results"),
            elementname=self._out_file_name + "_{}".format(poi_desc),
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
            poi_count = len(pois["Node"]) + len(pois["Way"]) + len(pois["Relation"])
            attractivity = calculate_attractivity(poi_count, weighting_factor)

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
                pois["Node"][poi_id]["attractivity"] = calculate_attractivity(
                    poi_area, weighting_factor
                )

            # Ways
            way_ids = list(map(int, pois["Way"].keys()))
            gd_ways = self._poi_data.loc[way_ids]
            gd_ways["area"] = gd_ways.geometry.area

            for id in gd_ways.index:
                poi_id = str(id)
                poi_area = gd_ways["area"].loc[id]

                pois["Way"][poi_id]["area"] = poi_area
                pois["Way"][poi_id]["attractivity"] = calculate_attractivity(
                    poi_area, weighting_factor
                )

            # Relations
            relation_ids = list(map(int, pois["Relation"].keys()))
            gd_relations = self._poi_data.loc[relation_ids]
            gd_relations["area"] = gd_relations.geometry.area

            for id in gd_relations.index:
                poi_id = str(id)
                poi_area = gd_ways["area"].loc[id]

                pois["Relation"][poi_id]["area"] = poi_area
                pois["Relation"][poi_id]["attractivity"] = calculate_attractivity(
                    poi_area, weighting_factor
                )

            return pois

    def _save(self, data):
        save_as_json_file(
            self._out_dir,
            self._out_file_name + ".json",
            data,
        )


def calculate_attractivity(attractivity, weighting_factor):
    return attractivity * weighting_factor


class PbfRoadNetworkFilter(Filter):
    def __init__(
        self,
        out_dir,
        out_file_name="PbfRoadNetworkFilter",
        make_tag_to_col=["maxspeed"],
        write_result=False,
    ) -> None:
        super().__init__(write_result=write_result, out_dir=out_dir)
        self._out_file_name = out_file_name
        self._make_tag_to_col = make_tag_to_col

    def filter(self, input_pbf_filepath) -> gpd.GeoDataFrame:
        osm = OSM(input_pbf_filepath)
        road_network = osm.get_network("all", extra_attributes=self._make_tag_to_col)

        return road_network

    def _save(self, data):
        save_gdf_to_geojson(data, self._out_dir, self._out_file_name)


class ConvertDataFrameColumnToString(Filter):
    def __init__(
        self,
        col_names: list,
        out_file_name: str = "ConvertDataFrameColumnToString",
        write_result: bool = False,
        out_dir: Union[str, None] = None,
    ) -> None:
        super().__init__(write_result=write_result, out_dir=out_dir)
        self._col_names = col_names
        self._out_file_name = out_file_name

    def _filter(self, data):
        for col_name in self._col_names:
            data[col_name] = data[col_name].astype(str)
            print(data["col_name"])
        return data

    def _save(self, data) -> None:
        save_gdf_to_geojson(data, self._out_dir, self._out_file_name)


class AddAltitudeToRoadNetwork(Filter):
    def __init__(self, out_dir, out_file_name, write_result=False) -> None:
        super().__init__(write_result=write_result, out_dir=out_dir)
        self._elevation_data = self._load_elevation_data()
        self._out_file_name = out_file_name

    def filter(self, road_network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        road_network["geometry"] = road_network["geometry"].apply(
            func=self._add_elevation_to_geom
        )
        return road_network

    def _load_elevation_data(self):
        cache_location = os.path.join(os.getcwd(), "data/cache/srtm")
        createDir(cache_location)
        elevation_data = srtm.get_data(local_cache_dir=cache_location)
        return elevation_data

    def _add_elevation(self, lat, lon):
        height = self._elevation_data.get_elevation(lat, lon)
        return (lat, lon, height)

    def _add_elevation_to_geom(self, geometry):
        transformed = shapely.ops.transform(self._add_elevation, geometry)
        return transformed

    def _save(self, data):
        save_gdf_to_geojson(data, self._out_dir, self._out_file_name)


class AddDefaultRoadNetAttributes(Filter):
    def __init__(
        self, out_dir, out_file_name, road_net_defaults_path, write_result=False
    ) -> None:
        super().__init__(write_result=write_result, out_dir=out_dir)
        self._out_file_name = out_file_name
        self._road_net_defaults_path = road_net_defaults_path

    def filter(
        self, road_network: Union[gpd.GeoDataFrame, pd.DataFrame]
    ) -> gpd.GeoDataFrame:
        """
        Derives and adds to an existing road network new column features with default
        values relevant to a Visum network.

        The default values and the new derived features are taken from configuration
        file which Visum uses to import an OSM network to translate it to a Visum
        network as explained in:
        https://cgi.ptvgroup.com/vision-help/VISUM_2021_DEU/Content/2_Schnittstellen/2_13_Aufbau_der_Konfigurationspakete.htm

        Args:
            road_network (GeoDataFrame or DataFrame): The road network data.

        Returns:
            A road network with Visum network related column features.
        """

        # network_defaults contains column features with default values relevant
        # to a visum network
        network_defaults = read_json_file(self._road_net_defaults_path)
        road_net_default_added = self._add_default_col_features(
            road_network, network_defaults
        )
        return road_net_default_added

    def _add_default_col_features(
        self,
        road_network: Union[gpd.GeoDataFrame, pd.DataFrame],
        net_defaults: dict,
    ) -> gpd.GeoDataFrame:
        """
        Derives and adds new column features with default values relevant to a
        Visum network to the existing road network.



        Args:
            road_network (DataFrame or GeoDataFrame): The road network data.
            net_defaults (dict): The default values to be added to the road network.

        Returns:
            A road network with new column features relevant to a Visum network.

        """
        # Fill road network with new attributes with default values
        road_net_default_added = road_network.assign(**net_defaults["DEFAULT"])

        key_val_unparsed_tag_tuples = [
            self._create_kv_unparsed_tag_tuple(unparsed_tag)
            for unparsed_tag in net_defaults.keys()
            if unparsed_tag != "DEFAULT"
        ]

        for tag, val, urban_net_defaults_key in key_val_unparsed_tag_tuples:
            if tag.lower() in road_net_default_added:
                # road_net.loc[road_net[key] == val, **]
                self._add_custom_default_col_features(
                    net_defaults,
                    road_net_default_added,
                    tag,
                    val,
                    urban_net_defaults_key,
                )

            else:
                # if tag doesn't exist as column in road_net_default_added,
                # then look in tag column if tag exists
                regex = '.("{}":"{}")'.format(tag, val)
                tag_exists_index = road_net_default_added["tags"].str.contains(
                    regex, regex=True
                )
                if tag_exists_index.any():
                    self._add_custom_default_col_features(
                        net_defaults,
                        road_net_default_added,
                        tag_exists_index,
                        val,
                        urban_net_defaults_key,
                    )

        return road_net_default_added

    def _add_custom_default_col_features(
        self, urban_net_defaults, road_net_default_added, tag, val, urban_net_defaults_key
    ):
        custom_default = urban_net_defaults[urban_net_defaults_key]

        for custom_default_attr, custom_default_attr_val in custom_default.items():
            road_net_default_added.loc[
                road_net_default_added[tag] == val, custom_default_attr
            ] = custom_default_attr_val

    def _create_kv_unparsed_tag_tuple(self, osm_kv_tag: str) -> tuple[str, str, str]:
        """
        Creates a three tuple from an unparsed OSM key-value tag.

        Args:
            osm_kv_tag (str): The OSM key-value tag in the form of "tag"="value".

        Returns:
            A three tuple consisting of (tag, value, unparsed osm key-value tag)
        """
        tag, val = self._parse_osm_kv_tag(osm_kv_tag)
        return (tag, val, osm_kv_tag)

    def _parse_osm_kv_tag(self, osm_tag):
        """
        Parse a string representing an OSM key value tag and return the key values as list.

        Args:
            condition (str): The OSM (key, value) tag where key is the tag.clear

        Returns:
            A list of form [tag, value].

        """
        ticks_removed = osm_tag.replace("'", "")
        double_quotes_removed = ticks_removed.replace('"', "")

        tag, value = double_quotes_removed.split("=")

        return [tag, value]

    def _save(self, data) -> None:
        save_gdf_to_geojson(data, self._out_dir, self._out_file_name)


class ValidateRoadNetwork(Filter):
    def __init__(
        self,
        out_dir,
        out_file_name="ValidateRoadNetwork",
        write_result=False,
    ) -> None:
        super().__init__(write_result=write_result, out_dir=out_dir)
        self._out_file_name = out_file_name

    def filter(self, road_network):
        has_neg_maxspeed = self._has_neg_maxspeed(road_network)
        has_neg_O2V_MAXSPEED = self._has_neg_o2v_maxspeed(road_network)

        self.print_error(has_neg_maxspeed, "Negative values for maxspeed found!")
        self.print_error(has_neg_O2V_MAXSPEED, "Negative O2V_MAXSPEED found!")
        return road_network

    def print_error(self, has_neg_maxspeed, error_msg):
        if has_neg_maxspeed:
            print(
                "\nWARNING! {}".format(error_msg)
                + "Please check the folder: <{}> to see in records with errors!\n".format(
                    self._out_dir
                )
            )

    def _has_neg_maxspeed(self, road_network, col_name="maxspeed"):
        road_net_without_na = road_network[road_network[col_name].notnull()]
        # print(road_network["maxspeed"].dropna().astype(int).head(15))
        records_with_neg_maxspeed = road_net_without_na[
            road_net_without_na[col_name].astype(int) < 0
        ]
        if len(records_with_neg_maxspeed.index) > 0:
            self._save_errors(
                out_dir=self._out_dir,
                file_name="records_with_negative_{}".format(col_name),
                records_w_errors=records_with_neg_maxspeed,
                col=["id", col_name],
            )
            return True
        else:
            return False

    def _has_neg_o2v_maxspeed(self, road_network):
        return self._has_neg_maxspeed(road_network, "O2V_MAXSPEED")

    def _save_errors(self, out_dir, file_name, records_w_errors, col=[]):
        errors = records_w_errors[col]
        save_gdf_to_csv(errors, out_dir, file_name)

    def _save(self, data) -> None:
        save_gdf_to_geojson(data, self._out_dir, self._out_file_name)


class Pipeline:
    def __init__(self, filters, data_sink: DataSink):
        self._filters = filters
        self._data_sink = data_sink

    def run(self, data_source: DataSource):
        if len(self._filters) == 0:
            raise NoStageDefinedError("No stage has been defined on this pipeline!")

        pipeline = self._create_pipeline(data_source=data_source, filters=self._filters)

        for result in pipeline:
            self._data_sink(result)

    def _create_pipeline(self, data_source: DataSource, filters):
        generator = data_source.read()

        for filter in filters:
            generator = filter(generator)
        return generator


def main():
    lie_out_poi_dir = os.path.join(os.getcwd(), "data/liechtenstein/poi")
    lie_roadnet_out_dir = os.path.join(os.getcwd(), "data/liechtenstein/road_network")
    lie_pbf_path = os.path.join(
        os.getcwd(), "data/osm/pbf_files/liechtenstein-140101.osm.pbf"
    )
    poi_filter_tags = Path(__file__).parents[0] / "resources" / "poi_filter_tags.json"
    poi_attractivity_info = os.path.join(
        os.getcwd(),
        "mobitopp_preprocessing/preprocessing/resources/poi_attractivity_info.csv",
    )
    lie_poi_path = "/Users/jibi/dev_projects/mobitopp/mobitopp-preprocessing/data/liechtenstein/pois/liechtenstein_poi.json"

    # Liechtenstein Pipeline
    """ lie_poi_pipeline = Pipeline(
        [
            PbfPoiFilter(lie_out_poi_dir, "liechtenstein_poi", poi_filter_tags),
            CalculateAttractivity(
                poi_attractivity_info_path=poi_attractivity_info,
                out_dir=lie_out_poi_dir,
                out_file_name="liechtenstein",
                pbf_path=lie_pbf_path,
                poi_filter_tags_path=poi_filter_tags,
            ),
        ]
    ) """
    # lie_poi_pipeline.run(lie_pbf_path)

    lie_road_filters = [
        PbfRoadNetworkFilter(lie_roadnet_out_dir, "lie_road_network", write_result=True),
        AddAltitudeToRoadNetwork(lie_roadnet_out_dir, "lie_road_net_with_elevation"),
        AddDefaultRoadNetAttributes(
            lie_roadnet_out_dir,
            "lie_road_net_with_defaults",
            URBAN_ROAD_NETWORK_DEFAULT,
        ),
        ValidateRoadNetwork(
            out_dir=lie_roadnet_out_dir,
            write_result=True,
        ),
    ]

    lie_road_pipeline = Pipeline(
        lie_road_filters, WriteGdfToFile(lie_roadnet_out_dir, "final_result")
    )

    lie_road_pipeline.run(FileDataSource(lie_pbf_path))
    """  pois_filter = PbfPoiFilter(lie_out_dir, "liechtenstein_poi", poi_filter_tags)
    pois = pois_filter.execute(lie_pbf_path)
    print(type(pois))

    calc_attract = CalculateAttractivity(
        poi_attractivity_info_path=poi_attractivity_info,
        out_dir=lie_out_dir,
        out_file_name="liechtenstein",
        pbf_path=lie_pbf_path,
        poi_filter_tags_path=poi_filter_tags,
        poi_path=lie_poi_path,
    )
    attract_pois = calc_attract.execute() """

    """ karlsruhe = "/Users/jibi/dev_projects/mobitopp/mobitopp-preprocessing/data/osm/pbf_files/karlsruhe-regbez-210614.osm.pbf"
    out_ka = os.path.join(os.getcwd(), "data/karlsruhe")

    ka_poi_filter = PbfPoiFilter(out_ka, "karlsruhe_poi", poi_filter_tags)
    ka_attractivity_filter = CalculateAttractivity(
        poi_attractivity_info_path=poi_attractivity_info,
        out_dir=out_ka,
        out_file_name="ka_attractivity",
        pbf_path=karlsruhe,
        poi_filter_tags_path=poi_filter_tags,
        use_cached_data=True,
    )

    ka_poi_pipeline = Pipeline([ka_poi_filter, ka_attractivity_filter])
    ka_poi_pipeline.run(karlsruhe) """


def road_net_pipeline():
    lie_pbf_path = os.path.join(
        os.getcwd(), "data/osm/pbf_files/liechtenstein-140101.osm.pbf"
    )

    geojson_path = "/Users/jibi/dev_projects/mobitopp/mobitopp-preprocessing/data/liechtenstein/road_network2/lie_road_network.geojson"
    lie_roadnet_out_dir = os.path.join(os.getcwd(), "data/liechtenstein/road_network2")

    lie_road_pipeline = Pipeline(
        [
            ConvertDataFrameColumnToString(
                ["length"], write_result=True, out_dir=lie_roadnet_out_dir
            ),
            AddAltitudeToRoadNetwork(lie_roadnet_out_dir, "lie_road_net_with_elevation"),
            AddDefaultRoadNetAttributes(
                lie_roadnet_out_dir,
                "lie_road_net_with_defaults",
                URBAN_ROAD_NETWORK_DEFAULT,
            ),
            ValidateRoadNetwork(
                out_dir=lie_roadnet_out_dir,
                write_result=True,
            ),
        ],
        WriteGdfToFile(lie_roadnet_out_dir, "final_result"),
    )

    lie_road_pipeline.run(GeoDataFrameStream(geojson_path, chunk_size=500))


if __name__ == "__main__":
    # main()
    road_net_pipeline()
