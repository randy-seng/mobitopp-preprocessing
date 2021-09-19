import os
import json

from pathlib import Path


def readTextFile(pathToTextFile):
    """Returns content of a file."""
    with open(pathToTextFile) as f:
        textFile = f.readlines()
        f.close()

    return textFile


def read_json_file(path):
    with open(path) as file:
        data = json.load(file)
        file.close()
        return data


def save_as_json_file(dir_path, filename, data):
    path = os.path.join(dir_path, filename + ".json")
    createFile(dir_path, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def is_file(file, file_type):
    return file.suffix.lower() == file_type


def file_exists(pathToFile):
    return Path(pathToFile).is_file()


def createFile(pathToDir, filename):
    # TODO: Remove this method and replace all usages of this method with create_file
    """Creates file if it doesn't exist."""
    file = os.path.join(pathToDir, filename)
    createDir(pathToDir)

    if not os.path.exists(file):
        open(file, "w").close()


def create_file(file_path):
    """Creates file if it doesn't exist."""
    dir_name = os.path.dirname(file_path)
    createDir(dir_name)

    if not os.path.exists(file_path):
        Path(file_path).touch(exist_ok=True)
        """ with open(file_path, "w") as file:
            file.close() """


def createDir(pathToDir):
    """Creates directory, if it doesn't exists."""
    Path(pathToDir).mkdir(parents=True, exist_ok=True)


def save_gdf_to_geojson(gdf, out_dir, out_file_name):
    """Save GeoDataFrame to out_dir/out_file_name.geojson"""
    path = os.path.join(out_dir, out_file_name + ".geojson")
    gdf.to_file(path, driver="GeoJSON")


def save_gdf_to_csv(gdf, out_dir, out_file_name):
    """Save GeoDataFrame to out_dir/out_file_name.csv"""
    path = os.path.join(out_dir, out_file_name + ".csv")
    gdf.to_csv(path)
