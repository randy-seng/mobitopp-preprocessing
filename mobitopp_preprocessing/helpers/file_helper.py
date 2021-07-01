from pathlib import Path
import os


def readTextFile(pathToTextFile):
    ''' Returns content of a file.

    '''
    with open(pathToTextFile) as f:
        textFile = f.readlines()
        f.close()

    return textFile


def fileAlreadyExists(pathToFile):
    return Path(pathToFile).is_file()


def createFile(pathToDir, filename):
    ''' Creates file if it doesn't exist.

    '''
    file = os.path.join(pathToDir, filename)
    createDir(pathToDir)

    if not os.path.exists(file):
        open(file, 'w').close()


def createDir(pathToDir):
    ''' Creates directory, if it doesn't exists.

    '''
    Path(pathToDir).mkdir(exist_ok=True)