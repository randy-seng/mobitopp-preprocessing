import urllib.request
import itertools
import os
import multiprocessing as mp
from datetime import datetime

from pathlib import Path

from mobitopp_preprocessing.helpers import file_helper
from mobitopp_preprocessing.helpers import url_helper
from mobitopp_preprocessing.helpers import md5_helper


def downloadFromURL(url, dirPath):
    """Download from url


    :param url: the url to download the data from
    :param dirPath: the path to the directory to save the data in
    :returns: True on SUCCESS, False on FAIL
    """
    fileName = url_helper.extractFileNameFromUrl(url)
    pathToFile = os.path.join(dirPath, fileName)

    # TODO: does dirPath only specify directory or the whole file
    if not url_helper.urlExist(url):
        print("URL is not working! Please check the url: {}".format(url))
        return False

    if file_helper.file_exists(pathToFile):
        # TODO: make method to extract the hash from an md5 file
        urlDotMD5File = url_helper.getDataFromUrl(url + ".md5")
        fileHash = md5_helper.createHash(pathToFile)
        urlHash = md5_helper.getHashFromMD5File(urlDotMD5File)
        # TODO: Check if it really gets an .md5 file containing the hash
        if md5_helper.hashesAreEqual(fileHash, urlHash):
            # file already downloaded
            print("File from {} already downloaded!".format(url))
            return False
        else:
            # TODO: Download and rename file as duplicate name exists
            newFileNamePath = appendTimestamp(pathToFile)
            url_helper.downloadFromUrl(url, newFileNamePath)
            return True
    else:
        # File not downloaded yet
        print("Downloading: " + fileName)
        urllib.request.urlretrieve(url, pathToFile)
        url_helper.downloadFromUrl(url, pathToFile)
        print("SUCCESS! {} downloaded!".format(fileName))
        return True


# TODO: Create test
def appendTimestamp(str):
    """


    :param str: a string
    :returns: the string appended with the current timestamp
    """
    date = datetime.now()
    timestamp = date.strftime("-%Y-%m-%dT%H:%M:%S")
    return str + timestamp


def downloadFromTextFileParallel(dirPath, pathToTextFile):
    textFile = file_helper.readTextFile(pathToTextFile)
    textWithoutLF = removeTrailingLF(textFile)
    argList = zip(textWithoutLF, itertools.repeat(dirPath))  # [(url, dirPath)]
    # Initialise and assign tasks to worker pool
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(downloadFromURL, argList)
    pool.close()


def removeTrailingLF(strList):
    return list(map(str.rstrip, strList))


def main():
    downloadLocation = Path(__file__).parents[2] / "data" / "osm" / "pbf_files"

    if not os.path.isdir(downloadLocation):
        os.makedirs(downloadLocation)

    # download Antarctica dataset
    # downloadFromURL(url, os.path.join(fileLocation, fileLocation))

    # download dataset from text file
    pathToTextFile = Path(__file__).parents[0] / "resources/download_urls.txt"
    # downloadFromTextFile(url, fileLocation, pathToFile)
    downloadFromTextFileParallel(downloadLocation, pathToTextFile)


if __name__ == "__main__":
    main()
