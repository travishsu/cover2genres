import urllib2
import pandas as pd
from util import getAlbumDetail

data_type = {"Filename": str, "Genres": str, "Release Year": int}

def addAlbum(query, datadir, idx=[0,0]):
    albums = pd.read_csv(datadir+"albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])
    detail = getAlbumDetail(query, idx=idx)

    filename = detail['title']+'_'+detail['release_date']
    u = urllib2.urlopen(detail['coverurl'])
    f = open(datadir+"img/"+filename+'.jpg', 'wb')
    f.write(u.read())
    f.close()

    newdata = pd.Series({"Filename": filename, "Genres": ' '.join(detail['genres']), "Release Year": detail['release_date']})
    if filename not in albums['Filename'].get_values():
        albums = albums.append(newdata, ignore_index=True)
        albums.to_csv(datadir+'albumlabel.csv', index=False)
    else:
        print('album already exists')
