import hashlib
from mobitopp_preprocessing.helpers import url_helper
from pathlib import Path


def hashesAreEqual(hash1, hash2):
    return hash1 == hash2


def createHash(pathToFile):
     with open(pathToFile, "rb") as f:
        file_hash = hashlib.md5()
        while chunk:= f.read(8192):
            file_hash.update(chunk)
        return file_hash.hexdigest()
    

def getHashFromMD5File(dotMd5File):
    return dotMd5File.split()[0] 


if __name__ == "__main__":
    """ password = "MD5Online"
    md5 = hashlib.md5(password.encode())
    
    print("The corresponding hash is : ")
    print(md5.hexdigest()) """
    urlMD5 = "http://download.geofabrik.de/south-america/suriname-latest.osm.pbf.md5" 
    url = "http://download.geofabrik.de/south-america/suriname-latest.osm.pbf" 

    downloadLocation = Path(__file__).parents[2] / "data" / "osm" / url_helper.extractFileNameFromUrl(url)
    print(downloadLocation)
    dotMD5File = url_helper.getDataFromUrl(urlMD5)
    hashOfFile = createHash(downloadLocation)
    hash = getHashFromMD5File(dotMD5File) 

    print(hashOfFile)
    print(hash)


