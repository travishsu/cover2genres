import spotipy
import urllib2
import discogs_client

s = spotipy.Spotify()
d = discogs_client.Client('ExampleApplication/0.1', user_token="tPgLfOQMObTxlKYXoSKpZkbVODRLZFqSwPzngIrb")
lowercase = 'abcdefghijklmnopqrstuvwxyz'
uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def toFilename(st):
    n = len(st)
    new = ""
    blank = False
    for i in xrange(n):
        s = st[i]

        if s == " ":
            blank = True
            continue
        if blank:
            new += s.upper()
            blank=False
        elif s in lowercase or s in uppercase:
            new += s
    return new

def transgen(s):
    new = ""
    for ss in s:
        if ss in lowercase:
            new += ss
    return new

def getFilename(detail):
    return toFilename(detail['album']) + "_" + toFilename(detail['artist'])

def getAlbumDetail(albumquery, qtype='album', idx = [0,0]):
    query = albumquery

    result = s.search(query, limit=1, type=qtype)['albums']['items'][idx[0]]
    albumid = result['id']
    albumname = result['name']
    try:
        a = [d.search(albumname, type='album')[idx[1]].data['genre'], d.search(albumname, type='album')[idx[1]].data['style']]
        genres = [gen.lower() for gen in a[0]]
        for genre in a[1]:
            genres.append(genre.lower())
    except:
        print('error')
        return None
    albumartist = result['artists'][0]['name']
    coverurl = result['images'][1]['url']
    coversize = (result['images'][1]['height']==300) and (result['images'][1]['width']==300)
    artistid = result['artists'][0]['id']
    artistgen = genres
    return {u'album'       : albumname,
            u'artist'      : albumartist,
            u'coverurl'    : coverurl,
            u'coversize'   : coversize,
            u'release_date': s.album(albumid)['release_date'],
            u'genres'      : [transgen(gen) for gen in artistgen]}
