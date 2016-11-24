import spotipy
import urllib2
import pandas as pd
from util import getAlbumDetail, getFilename

data_type = {"Filename": str, "Genres": str, "Release Year": int}
albums = pd.read_csv("albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])

def addAlbum(query, dir='data/'):
    albums = pd.read_csv("albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])
    detail = getAlbumDetail(query)
    if detail['coversize']==False:
        print("unmatched pixel size")
        return None
    filename = getFilename(detail)
    u = urllib2.urlopen(detail['coverurl'])
    f = open(dir+filename+'.jpg', 'wb')
    f.write(u.read())
    f.close()
    newdata = pd.Series({"Filename": filename, "Genres": ' '.join(detail['genres']), "Release Year": detail['release_date']})
    if filename not in albums['Filename'].get_values():
        albums = albums.append(newdata, ignore_index=True)
        albums.to_csv('albumlabel.csv', index=False)
    else:
        print('album already exists')
