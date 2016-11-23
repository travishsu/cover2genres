from PIL import Image

uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
dirname = "data/"
filename = "Master_of_Puppets_Metalica.jpg"
filename = "AmericanIdiot_GreenDay.jpg"
im = Image.open(dirname+filename)
print(im.size)
pix = im.load()

def formalString(st):
    n = len(st)
    new = ""
    for i in xrange(n):
        s = st[i]
        if s not in uppercase or i < 1:
            new += s
        else:
            new += (" "+s)
    return new

class Album:
    def __init__(self, filename):
        albumname_artist = str.split(filename, ".")[0]
        albumname_artistsplit = str.split(albumname_artist, "_")
        self.albumname = formalString(albumname_artistsplit[0])
        self.artist    = formalString(albumname_artistsplit[1])
        self.labels = set()
        im = Image.open(dirname+filename)
        self.coversize = im.size
        pix = array(im.getdata()).reshape(300, 300, 3)
        self.pix = array([ [ [ pix[j][k][i] for k in xrange(300)] for j in xrange(300)] for i in xrange(3)])

    def getDetail(self):
        return {"AlbumName": self.albumname,
                "Artist"    : self.artist,
                "Labels"    : self.labels,
                "Pixel"     : self.pix,
                "PixelSize" : self.coversize}
    def addLabel(self, label):
        if label not in self.labels:
            self.labels.add(label)
        else:
            print("=== Label already exists. ===")

    
class Albumbase:
    def __init__(self):
        self.nameindex = set()
        self.albums    = dict()
