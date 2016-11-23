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
