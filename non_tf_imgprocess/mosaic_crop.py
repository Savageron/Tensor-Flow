# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 03:42:04 2017

@author: Jin Lim

To crop items of interest in an annotated mosaic. Coordinates of each
item should be in a csvfile, with column 1 = x coordinate, column 2 = y coordinate.
The mosaic is to be read from the top left corner, ie. top left corner pixel
coordinate = (1, 1). x coordinate increases from left to right, y coordinate 
increase from top to bottom of the mosaic.

Parameters to define are: 
excelfile - name (with path) of csvfile
mosfile - name (with path) of mosaic file
imgsize - dimensions of desired cropped image (eg. if 24x24 pixels, imgsize = 24)
imgname - naming of the item of interest (without .tif) #update!
path - path of folder to save cropped images
trainingtest - ratio of train to test
#add species....
....in complete**!
*(with path) - only if this python file is saved in a different folder
*increase the number "5" in "%5d" if more number is used

tasks to do
make the code automatically create a folder that contain folders of 
images with repective species names nicely done, automatically in code.
make sure the trainingtest will make the ratio into a nice number so user can 
just decide a ratio and not worry about the code not running
"""

#import os
#import numpy as np
import csv
from PIL import Image
import random
import numpy

Image.MAX_IMAGE_PIXELS = 1000000000 #to silence DecompressionBombWarning DOS

#species = 2
#if species = 2 means Bathy....

#should i use a,b,c,d,e,f instead of 0,1,2,3...?
Alv = ["Alvinocaridid", "Alv", "Alvinocaridid", 3]
BJa = ["Bathymodiolus_japonicus", "BJa", "Bathymodiolus_japonicus", 0] 
BPl = ["Bathymodiolus_platifrons_corrected", "BPl", "Bathymodiolus_platifrons", 1]
Par = ["Paralomis", "Par", "Paralomis", 4]
SCr = ["Shinkaia_crosnieri", "SCr", "Shinkaia_crosnieri", 2]
TDe = ["Thermosipho_desbruyesi", "TDe","Thermosipho_desbruyesi", 5]

"""to automate run all species"""
#x = [Alv, BJa, BPl, Par, SCr, TDe]
#for i in x:
#    species = i
#    training(trainingtest)

species = SCr

excelfile = "{}.csv".format(species[0])
mosfile = "D:/Iheya_n/NBC_2m_2014/mosaics/NBC_2m_2014_mosaic_layer.tif"
imgsize = 50
imgname = species[1]
path = "D:/Iheya_n/NBC/{}/{}_{}".format(species[2], species[1], imgsize)
"""check if cifar code already included training and testing sample"""
trainingtest = 3/1
totalnumber = 7000 #"""number of each species ie. test + training"""

#counts the number of items of interest in the csvfile
def count():
    m = 0
    with open (excelfile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            m = m+1
    return m

#crops and saves all the sample images in a species
def mosaic_crop(excelfile, mosfile, imgsize, imgname): 
    img = Image.open(mosfile)
    with open (excelfile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        n = 0
        for row in readCSV:
            img1 = img.crop(
                    (
                            float(row[0])-imgsize/2,
                            float(row[1])-imgsize/2,
                            float(row[0])+imgsize/2,
                            float(row[1])+imgsize/2
                    )
            )
            n = n+1
            img1.save(path + "/" + imgname + "_{}_all/{}-{}.tif".\
                      format(imgsize, imgname, "%05d" % n))

#generate training and test set, training and test csv file
def training(trainingtest):
    trainingnumber = int(trainingtest * totalnumber / (trainingtest + 1))
    trainingset = numpy.zeros(shape=(trainingnumber, 1))
    testset = numpy.zeros(shape=(totalnumber-trainingnumber, 1))
    j = 0
    for i in random.sample(range(1, (count()+1)), totalnumber): #python2 use xrange
        if j < trainingnumber:
            trainingset[j]="%5d" % i
            j=j+1
        else:
            testset[j-trainingnumber]="%5d" % i
            j=j+1
    numpy.savetxt(path + "/" + imgname + "train.csv", trainingset, delimiter=",")
    numpy.savetxt(path + "/" + imgname + "test.csv", testset, delimiter=",")

#def convertImg():
    ImgArrayTrain = open(path + "/{}train_{}.txt".format(species[1], imgsize), "a")
    ImgArrayTest = open(path + "/{}test_{}.txt".format(species[1], imgsize), "a")
    for i in trainingset:
        Io = Image.open(path + "/" + imgname + "_{}_all/{}-{}.tif".\
                        format(imgsize, imgname, "%05d" % i)) #"%05d" % n)
        Iar = numpy.asarray(Io)
        Iar1 = "{},".format(species[3])
        for row in Iar:
            for element in row:
                Iar1 = Iar1 + str(element[0]) + ","
        for row in Iar:
            for element in row:
                Iar1 = Iar1 + str(element[1]) + ","
        for row in Iar:
            for element in row:
                Iar1 = Iar1 + str(element[2]) + ","
#or use for loop, then str(Iar[0,0,i]), or [0,r,i]...??
#        Iar1 = str(Iar.tolist())
#        lineToWrite = species[3] +":"+Iar1+"\n"
#        ImgArrayTrain.write(lineToWrite)
        ImgArrayTrain.write(Iar1)
    for i in testset:
        Io = Image.open(path + "/" + imgname + "_{}_all/{}-{}.tif".\
                        format(imgsize, imgname, "%05d" % i)) #"%05d" % n)
        Iar = numpy.asarray(Io)
        Iar1 = "{},".format(species[3])
        for row in Iar:
            for element in row:
                Iar1 = Iar1 + str(element[0]) + ","
        for row in Iar:
            for element in row:
                Iar1 = Iar1 + str(element[1]) + ","
        for row in Iar:
            for element in row:
                Iar1 = Iar1 + str(element[2]) + ","
#        Iar1 = str(Iar.tolist())
#        lineToWrite = species[3] +":"+Iar1+"\n"
#        ImgArrayTest.write(lineToWrite)
        ImgArrayTest.write(Iar1)

#currently this function for training only for first attempt. just so happen training
#ratio is just nice, reconstruct code for other training ratio, esp folder organisation
#now u know how to create file, create folders too.
def txttobinconverter():
    testortrain = ['train'] #, 'test']
    for i in testortrain:
        readtxt = open(path + "/{}{}_{}.txt".format(species[1], i, imgsize), "r")
        string = readtxt.read()
        stringtolist = [int(s) for s in string[:-1].split(",")]
        buffer = bytes(stringtolist)
        strfile = r"{}/{}{}_{}.bin".format(path, species[1], i, imgsize)
        with open(strfile, 'bw') as f:
            f.write(buffer)

def txttobinconvertertestcombine():
    selectedspecies = [BJa, BPl, SCr]
    string = ""
    for i in selectedspecies:
        species = i
        path = "D:/Iheya_n/NBC/{}/{}_{}".format(species[2], species[1], imgsize)
        readtxt = open(path + "/{}{}_{}.txt".format(species[1], 'test', imgsize), "r")
        string = string + readtxt.read()
    stringtolist = [int(s) for s in string[:-1].split(",")]
    buffer = bytes(stringtolist)
    strfile = r"{}/{}{}_{}.bin".format(path[:10], "BJaBPlSCr", 'test', imgsize)
    with open(strfile, 'bw') as f:
        f.write(buffer)

#mosaic_crop(excelfile, mosfile, imgsize, imgname)
#training(trainingtest)
#txttobinconverter()

#my initial method of saving training and test set individually, waste of space
#"""
#import numpy
#from shutil import copy
#import random
#
##def ... for variable 70x70 or 50 etc...
#k = int(7160/2)  #read length and divide by 2. option to choose how many percent data too training
#a = numpy.zeros(shape=(k,1))
##random.sample(xrange(1,7160, 7160/2)) #for python 2 use xrange
#j=0;
#for i in random.sample(range(1, 7161), k):
#    //copy("D:\Shinkaia_crosnieri\S_C_70x70\S_C_img_70x70\img{}.tif".format(i), \
#    //     "D:\Shinkaia_crosnieri\S_C_70x70\S_C_img_70x70_0.5")
#    a[j]=i
#    j=j+1
#
#numpy.savetxt("D:\Shinkaia_Crosnieri\S_C_70x70\S_C_data_70x70_0.5.csv", a, \
#              delimiter=",") 
#"""
#
##Dr Blair's modification of code to save just the numbers
#"""
#import numpy
##from shutil import copy
#import random
#
##def ... for variable 70x70 or 50 etc...
#k = int(7160/2)  #read length and divide by 2. option to choose how many percent data too training
#a = numpy.zeros(shape=(k,1))
#b = numpy.zeros(shape=(k,1))
##random.sample(xrange(1,7160, 7160/2)) #for python 2 use xrange
#j=0;
#for i in random.sample(range(1, 7161), 7160):
##    //copy("D:\Shinkaia_crosnieri\S_C_70x70\S_C_img_70x70\img{}.tif".format(i), \
##    //     "D:\Shinkaia_crosnieri\S_C_70x70\S_C_img_70x70_0.5")
#    if j<k:
#        a[j]=i
#        j=j+1
#    else:
#        b[j-k]=i
#        j=j+1
#
#numpy.savetxt("D:\Shinkaia_Crosnieri\S_C_70x70\S_C_data_70x70_0.5_training.csv", a, \
#              delimiter=",")
#numpy.savetxt("D:\Shinkaia_Crosnieri\S_C_70x70\S_C_data_70x70_0.5_test.csv", b, \
#              delimiter=",")
#"""
#
#"""old code"""
#    with open('Shinkaia_crosnieri.csv') as csvfile:
#            readCSV = csv.reader(csvfile, delimiter=',')
#            xcoor = []
#            ycoor = []
#            for row in readCSV:
#                x = row[0]
#                y = row[1]
#                xcoor.append(x)
#                ycoor.append(y)
#            xarray = np.asarray(xcoor)
#            yarray = np.asarray(ycoor)
#            xfloat = xarray.astype(np.float)
#            yfloat = yarray.astype(np.float)
#            print(xfloat) #1-7160, 0-7159
#            print(yfloat) #1-7160, 0-7159

#img = Image.open("NBC_2m_2014_mosaic_layer.tif")

##def ...
#for i in range (0, 7160):
#    img1 = img.crop( #70x70pixel pic
#            (
#                    xfloat[i]-25, 
#                    yfloat[i]-25, 
#                    xfloat[i]+25,
#                    yfloat[i]+25
#            )
#    )
#    img1.save("D:\Shinkaia_crosnieri\S_C_50x50\S_C_img_50x50\img{}.tif".format(i+1))

#    filename = "img{}.tif".format(i+1)
#    path = ":%labels\Shinkaia_crosnieri_images"
#    path = "D:\Shinkaia_crosnieri_images"
#    fullpath = os.path.join(path, filename)
#    f = open(fullpath, "a")
#    f = open(fullpath, "w")
#    f.write(img1)
#f.close()
#    img1.save(filename)
#    img2 = img.crop((0, 0, 100, 100))
#    img2.save("img{}.jpg")