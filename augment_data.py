from scipy.misc import imread, imsave, imrotate
import pandas as pd
import subprocess

setpath = 'data/set2/'
csvpath = setpath + 'albumlabel_aug.csv'

subprocess.call(['cp', '-r', setpath+'resize', setpath+'augmented'])
subprocess.call(['cp', setpath+'albumlabel.csv', setpath+'albumlabel_aug.csv'])

data = pd.read_csv(csvpath, dtype={"Filename": str, "Genres": str, "Release Year": int},
                   parse_dates=["Release Year"])
N = data.shape[0]

for i in range(N):
    row = data.ix[i]
    im = imread(setpath+'augmented/'+row.Filename+'.jpg')
    for angle in [90, 180, 270]:
        new = row.copy()
        new.Filename = new.Filename + '_' + str(angle)
        data = data.append(new)
        rotated_im = imrotate(im, angle)
        new_file = setpath+'augmented/'+new.Filename+'.jpg'
        imsave(new_file, rotated_im)

data.to_csv(csvpath, index=False)
