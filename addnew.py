import urllib2
import pandas as pd
from util import getAlbumDetail, getAlbumDetailByID, replace, repairhead
import discogs_client

d = discogs_client.Client('ExampleApplication/0.1', user_token="tPgLfOQMObTxlKYXoSKpZkbVODRLZFqSwPzngIrb")
data_type = {"Filename": str, "Genres": str, "Release Year": int}

def addAlbum(query, datadir, idx=[0,0]):
    albums = pd.read_csv(datadir+"albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])
    detail = getAlbumDetail(query, idx=idx)

    filename = detail['title']+'_'+detail['release_date']
    if detail['coverurl'] == u'':
        print('no image provided')
        return None
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

def addAlbumByID(rid, datadir, title=None):
    if rid==None:
        return None
    albums = pd.read_csv(datadir+"albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])
    detail = getAlbumDetailByID(rid)

    if title != None:
        filename = repairhead(replace(title)) +'_' + detail['release_date']
    elif detail['title'] == None:
        filename = detail['title']+'_'+detail['release_date']
    else:
        return 'Title is None'
    if detail['coverurl'] == u'':
        print('no image provided')
        return None

    u = urllib2.urlopen(detail['coverurl'])
    f = open(datadir+"img/"+filename+'.jpg', 'wb')
    f.write(u.read())
    f.close()

    newdata = pd.Series({"Filename": filename, "Genres": ' '.join(detail['genres']), "Release Year": detail['release_date']})
    if filename not in albums['Filename'].get_values():
        albums = albums.append(newdata, ignore_index=True)
        albums.to_csv(datadir+'albumlabel.csv', index=False)
    else:
        return ' |--- [Already exists]'

def addAlbumByLabelID(label_id, setpath, limit=100, n_version=100):
        l = d.label(label_id)
        #lst = [k for k in l.releases if type(k) == discogs_client.models.Master]
        lst = l.releases
        count = 0
        for i in xrange(limit):
            try:
                text = addAlbumByFilterMaster(lst[i].master, setpath, n_version)
                print "[OK  ] Progress: {}/{}, {}".format(i+1, limit, lst[i].title.encode('ascii', 'ignore')),
                print(text)
                if 'Success' in text:
                    count += 1
            except KeyboardInterrupt:
                break
            except:
                print("[Fail] Progress: {}/{}".format(i+1, limit))
        print("Summary: {} data added".format(count))

def addAlbumByArtistID(artist_id, setpath, limit=100, n_version=100):
        l = d.artist(artist_id)
        #lst = [k for k in l.releases if type(k) == discogs_client.models.Master]
        lst = l.releases
        count = 0
        for i in xrange(limit):
            try:
                text = addAlbumByFilterMaster(lst[i], setpath, n_version)
                print "[OK  ] Progress: {}/{}, {}".format(i+1, limit, lst[i].title.encode('ascii', 'ignore')),
                print(text)
                if 'Success' in text:
                    count += 1
            except KeyboardInterrupt:
                break
            except:
                print("[Fail] Progress: {}/{}".format(i+1, limit))
        print("Summary: {} data added".format(count))

def addAlbumByFilterMaster(release, setpath, n_version):
    if release.main_release == None:
        text = ' |--- [Not a master]'
        return text
    n = release.versions.count
    if n < n_version:
        text = ' |--- [Number of versions too little: {}]'.format(n)
        return text
    rid = release.main_release.id
    text = addAlbumByID(rid, setpath, title=release.title)
    if text==None:
        text = ' |--- [Success]'
    return text
