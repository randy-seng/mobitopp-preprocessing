import shutil
import os

import pytest
from pathlib import Path

from mobitopp_preprocessing.acquire import geofabrik_osm_data as od


@pytest.fixture()
def path():
    path = Path(__file__).parents[1] / "resources" / "poi-data"
    os.makedirs(name=path, exist_ok=True)
    yield path
    shutil.rmtree(path)


def test_downloadFromLink(path):
    url = "https://download.geofabrik.de/antarctica-latest.osm.pbf"
    fileName = url.split("/")[-1]
    location = os.path.join(path, fileName)
    assert "peter" == str(location)
    print(str(location))

    od.downloadFromURL(url, location)
