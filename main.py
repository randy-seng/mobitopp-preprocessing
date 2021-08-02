from mobitopp_preprocessing.acquire import geofabrik_osm_data
from mobitopp_preprocessing.preprocessing import pipeline
from icecream import install

install()

if __name__ == "__main__":
    print("Starting!")
    pipeline.main()
