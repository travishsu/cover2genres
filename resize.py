from scipy.misc import imread, imsave, imresize
import glob
import subprocess

setpath = 'data/set1/'
subprocess.call(['cp', '-r', setpath+'img', setpath+'resize'])

for pic in glob.glob(setpath+"resize/*.jpg"):
    img = imread(pic)
    img = imresize(img, (128, 128))
    imsave(pic, img)
