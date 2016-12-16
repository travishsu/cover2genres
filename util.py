import discogs_client

d = discogs_client.Client('ExampleApplication/0.1', user_token="tPgLfOQMObTxlKYXoSKpZkbVODRLZFqSwPzngIrb")
lowercase = 'abcdefghijklmnopqrstuvwxyz'
uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def transgen(s):
    new = ""
    for ss in s:
        if ss in lowercase:
            new += ss
    return new

def getAlbumDetail(albumquery, qtype='release', idx = [0,0]):
    result = d.search(albumquery, type=qtype)[0].data
    a = [result['genre'], result['style']]
    genres = [gen.lower() for gen in a[0]]
    for genre in a[1]:
        genres.append(genre.lower())

    return {u'title'         : result['title'],
            u'coverurl'      : result['thumb'],
            u'release_date'  : result['year'],
            u'genres'        : [transgen(gen) for gen in genres]}

def getAlbumDetailByID(rid):
    result = d.release(rid)
    a = [result.genres, result.styles]
    genres = [gen.lower() for gen in a[0]]
    if a[1]!=None:
        for genre in a[1]:
            genres.append(genre.lower())

    return {u'title'         : result.title,
            u'coverurl'      : result.thumb,
            u'release_date'  : str(result.year),
            u'genres'        : [transgen(gen) for gen in genres]}
