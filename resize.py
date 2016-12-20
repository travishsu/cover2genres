from scipy.misc import imread, imsave, imresize
import glob
import subprocess

setpath = 'data/set2/'
subprocess.call(['cp', '-r', setpath+'img', setpath+'resize'])

for pic in glob.glob(setpath+"resize/*.jpg"):
    img = imread(pic)
    img = imresize(img, (32, 32))
    imsave(pic, img)
