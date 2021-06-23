from pathlib import Path

def readTextFile(pathToTextFile):
    with open(pathToTextFile) as f:
        textFile = f.readlines()
        f.close()

    return textFile


def fileAlreadyExists(pathToFile):
    return Path(pathToFile).is_file()

