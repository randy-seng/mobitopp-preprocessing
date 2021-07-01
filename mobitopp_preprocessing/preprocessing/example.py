import configparser, contextlib
import os, sys, json
import pandas as pd
from pathlib import Path
from esy.osmfilter import osm_colors as CC
from esy.osmfilter import run_filter
from esy.osmfilter import Node, Way, Relation
from esy.osmfilter import export_geojson
from mobitopp_preprocessing.helpers import file_helper


if __name__ == "__main__":
    print("Current working directory is {}".format(os.getcwd()))
    PBF_inputfile = os.path.join(os.getcwd(), "data/osm/pbf_files/liechtenstein-140101.osm.pbf")
    JSON_outputfile = os.path.join(os.getcwd(), "data/osm/json/liechtenstein/liechtenstein_POI.json")
    taglist_json = Path(__file__).parents[0] / "resources"/ "poi_taglist.json"

    # Read in taglist file for filtering
    f = open(taglist_json)
    taglist = json.load(f)
    f.close()

    # Create prefilter
    """ prefilter = {
        Node: {"amenity": [True]},
        Way: {"amenity": [True]},
        Relation: {"amenity": [True]}
    } """

    prefilter = {
        Node: taglist,
        Way: taglist,
        Relation: taglist
    }

    whitefilter = [
        [("amenity", "restaurant")]
    ]

    blackfilter = [()]
    print(blackfilter)
    print(whitefilter)

    """ whitefilter = [(("waterway", "drain"), ("name", "Wäschgräble")), ]
    blackfilter = [("pipeline", "substation")] """

    [Data, Elements] = run_filter(
            'POI',
            PBF_inputfile,
            JSON_outputfile,
            prefilter,
            whitefilter,
            blackfilter,
            NewPreFilterData=True,
            CreateElements=True,  # initiate main filter phase
            LoadElements=False,
            verbose=True
        )
    
    """ print(len(Data["Node"]))
    print(len(Data["Relation"]))
    print(len(Data["Way"]))
    print("\n")

    print(len(Elements['POI']['Node']))
    print(len(Elements['POI']['Relation']))
    print(len(Elements['POI']['Way']))
    file_helper.createFile("data/osm/geojson/", "test.geojson")
    export_geojson(Elements["POI"]["Way"], Data, filename="data/osm/geojson/test.geojson", jsontype="Line") """
    df = pd.DataFrame(Data)
    print(df.head())
