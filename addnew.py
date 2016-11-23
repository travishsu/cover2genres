import pandas as pd
from util import *

data_type = {"Filename": str, "Genres": str, "Release Year": int}
albums = pd.read_csv("albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])

def getFilename(detail):
    return toFilename(detail['album']) + "_" + toFilename(detail['artist'])

def getAlbumDetail(query):
    query = 'album:'+query
    albumid = s.search(query, limit=1)['tracks']['items'][0]['album']['id']
    albumname = s.search(query, limit=1)['tracks']['items'][0]['album']['name']
    albumartist = s.search(query, limit=1)['tracks']['items'][0]['album']['artists'][0]['name']
    coverurl = s.search(query, limit=1)['tracks']['items'][0]['album']['images'][1]['url']
    coversize = (s.search(query, limit=1)['tracks']['items'][0]['album']['images'][1]['height']==300) and (s.search(query, limit=1)['tracks']['items'][0]['album']['images'][1]['width']==300)
    artistid = s.search(query, limit=1)['tracks']['items'][0]['album']['artists'][0]['id']
    artistgen = s.artist(artistid)['genres']
    return {u'album':    albumname,
            u'artist':   albumartist,
            u'coverurl': coverurl,
            u'coversize': coversize,
            u'release_date': s.album(albumid)['release_date'],
            u'genres':   [transgen(gen) for gen in artistgen]}


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
