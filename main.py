import mobitopp_preprocessing
from mobitopp_preprocessing.acquire import osm_data as osm
from pathlib import Path


def main():
    url = "https://download.geofabrik.de/antarctica-latest.osm.pbf"
    downloadLocation = Path(__file__).parents[0] / "data" / "osm"

    # download Antarctica dataset
    # downloadFromURL(url, os.path.join(downloadLocation, downloadLocation))

    # download dataset from text file
    pathToFile =Path(__file__).parents[0] / "mobitopp_preprocessing" / "acquire" / "resources" / "download_urls.txt"
    # downloadFromTextFile(url, downloadLocation, pathToFile)
    osm.downloadFromTextFileParallel(url, downloadLocation, pathToFile)


if __name__ == "__main__":
    main()
