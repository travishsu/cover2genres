import spotipy
import urllib2

s = spotipy.Spotify()
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
        else:
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

def getAlbumDetail(albumquery, qtype='album'):
    query = 'album:'+albumquery

    result = s.search(query, limit=1, type=qtype)['albums']['items'][0]
    albumid = result['id']
    albumname = result['name']
    albumartist = result['artists'][0]['name']
    coverurl = result['images'][1]['url']
    coversize = (result['images'][1]['height']==300) and (result['images'][1]['width']==300)
    artistid = result['artists'][0]['id']
    artistgen = s.artist(artistid)['genres']
    return {u'album'       : albumname,
            u'artist'      : albumartist,
            u'coverurl'    : coverurl,
            u'coversize'   : coversize,
            u'release_date': s.album(albumid)['release_date'],
            u'genres'      : [transgen(gen) for gen in artistgen]}
