import requests 
import urllib.request


def extractFileNameFromUrl(url):
    return url.split('/')[-1]


def getDataFromUrl(url):
    response = requests.get(url)
    return response.text


def urlExist(url):
    status_code = requests.head(url).status_code
    if (status_code < 400):
        return True
    else:
        return False 


def downloadFromUrl(url, pathToFile):
    urllib.request.urlretrieve(url, pathToFile)
