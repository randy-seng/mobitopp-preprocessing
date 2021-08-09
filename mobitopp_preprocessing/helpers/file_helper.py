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
    path = os.path.join(dir_path, filename)
    createFile(dir_path, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def is_file(file, file_type):
    return file.suffix.lower() == file_type


def file_exists(pathToFile):
    return Path(pathToFile).is_file()


def createFile(pathToDir, filename):
    """Creates file if it doesn't exist."""
    file = os.path.join(pathToDir, filename)
    createDir(pathToDir)

    if not os.path.exists(file):
        open(file, "w").close()


def createDir(pathToDir):
    """Creates directory, if it doesn't exists."""
    print(Path(pathToDir).is_dir())
    Path(pathToDir).mkdir(parents=True, exist_ok=True)
