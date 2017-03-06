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
boxsize - dimensions of desired cropped image (eg. if 24x24 pixels, boxsize = 24)
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
#import copy
#import time
import sys

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
NSe = ["Null_set", "NSe", "Null_set", 6]

"""to automate run all species"""
#x = [Alv, BJa, BPl, Par, SCr, TDe]
#for i in x:
#    species = i
#    training(trainingtest)

allspecies = [Alv, BJa, BPl, Par, SCr, TDe]
species = SCr

excelfile = "{}.csv".format(species[0])
mosfile = "D:/Iheya_n/NBC_2m_2014/mosaics/NBC_2m_2014_mosaic_layer.tif"
boxsize = 50  #must be even number, boxsize of 50 means 50x50pixel box
imgname = species[1]
path = "D:/Iheya_n/NBC/{}/{}_{}".format(species[2], species[1], boxsize)
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

#crops and saves all the sample images in a species ***can save space by directly converting to array and bin file, just csv file for reference instead of saving the images.
def mosaic_crop(excelfile, mosfile, boxsize, imgname): 
    img = Image.open(mosfile)
    with open(excelfile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        n = 0
        for row in readCSV:
            img1 = img.crop(
                    (
                            float(row[0])-boxsize/2,
                            float(row[1])-boxsize/2,
                            float(row[0])+boxsize/2,
                            float(row[1])+boxsize/2
                    )
            )
            n = n+1
            img1.save(path + "/" + imgname + "_{}_all/{}-{}.tif".\
                      format(boxsize, imgname, "%05d" % n))

#20320x20320 pixels, include line if array all white (ie [255,255,255]) or more than 75% white then delete it
#for 50x50 pixel of whole 20320x20320 map, 20 pixel at the right end and btm end neglected
#generate random number and filter, and ... 
def null_set():
    nullpath = path[:14]+"/{}/{}_{}".format(NSe[0],NSe[1],boxsize)
    nullList = []
    img = Image.open(mosfile)
#    pixel = [-boxsize/2, -boxsize/2]
    numberofhorizontalrows = int(img.size[0]/boxsize)
    numberofverticalcolumns = int(img.size[1]/boxsize)
#    imgwithallspecies0 = []
#    for i in range(numberofhorizontalrows*50): #took too long to generate 412090000 coordinates
#        for j in range(numberofverticalcolumns*50):
#            pixel = [1 + i*1, 1 + i*j]
#            imgwithallspecies0.append(pixel)
    for j in range(numberofhorizontalrows):
        for i in range(numberofverticalcolumns):
            pixel = [boxsize/2 + i*boxsize, boxsize/2 + j*boxsize]
            nullList.append(pixel)
    allspeciescoord = []
    for i in allspecies:
        species = i
        excelfile = "{}.csv".format(species[0])
        with open(excelfile) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            for row in readCSV:
                coord = [float(row[0]),float(row[1])]
                allspeciescoord.append(coord)
    randomnullListtemp = random.sample(range(0,len(nullList)), len(nullList))
#    randomnullListtemp = copy.copy(nullList)
    randomnullListtemp2 = []
    r = -1
    n = 0
    while len(randomnullListtemp2) < totalnumber: #problem: is the first 7000 of the random sample 164836 truly randomly distributed?
        r = r + 1
        rndmcoord = nullList[randomnullListtemp[r]]
        coordinanimalbox = 1
        for i in allspeciescoord:
#            coordinanimalbox = 1 #if the the coord is within crab box value = 0
            if abs(rndmcoord[0]-i[0]) < boxsize + 25:
                if abs(rndmcoord[1]-i[1]) < boxsize + 25:
                    coordinanimalbox = coordinanimalbox * 0
                    break
        if coordinanimalbox == 1:
#    print (len(randomnullListtemp2)
            img1 = img.crop(
                    (
                            rndmcoord[0]-boxsize/2,
                            rndmcoord[1]-boxsize/2,
                            rndmcoord[0]+boxsize/2,
                            rndmcoord[1]+boxsize/2
                    )
            )
            Iar = numpy.asarray(img1)
            Iar2 = Iar.tolist()
            numberofwhitepixels = 0
            for row in Iar2:
                numberofwhitepixels = numberofwhitepixels + \
                row.count([255, 255, 255])
            if numberofwhitepixels/boxsize**2 < 0.25:
                randomnullListtemp2.append(rndmcoord)
                n = n + 1
                img1.save(nullpath+"/{}_{}_all/{}-{}.tif".format\
                          (NSe[1],boxsize,NSe[1],"%05d"%n))
    nullArray = numpy.asarray(randomnullListtemp2)
    numpy.savetxt(nullpath+"/{}_{}.csv".format(NSe[0],boxsize),\
                  nullArray, delimiter=",")

#    for i in nullList: #this step is taking too long!
#        for j in allspeciescoord:
#            if abs(i[0]-j[0]) < boxsize + 25: #assuming all creatures lies within boxsize + 25 pixels
##            if i[0]-boxsize/2<= j[0]<= i[0]+boxsize/2:
#                if abs(i[1]-j[1]) < boxsize + 25:
##                if i[1]-boxsize/2<= j[1]<= i[1]+boxsize/2:
#                    nullList.remove(i)
#        n = 0
#        for j in nullList:
#            img1 = img.crop(
#                    (
#                            j[0]-boxsize/2,
#                            j[1]-boxsize/2,
#                            j[0]+boxsize/2,
#                            j[1]+boxsize/2
#                    )
#            )
#            Iar = numpy.asarray(img1)
#            Iar2 = Iar.tolist()
#            numberofwhitepixels = 0
#            for row in Iar2:
#                numberofwhitepixels = numberofwhitepixels + \
#                row.count([255, 255, 255])
#            if numberofwhitepixels/boxsize**2 > 0.25:
#                nullList.remove(j)
#            else:
#                n = n+1
#                img1.save(nullpath+"/{}_{}_all/{}-{}.tif".format(NSe[1],\
#                          boxsize,NSe[1],"%06d"%n))
#    nullArray = numpy.asarray(nullList)
#    numpy.savetxt(nullpath+"/{}_{}.csv".format(NSe[0],boxsize),\
#                      nullArray, delimiter=",")

#        path = "D:/Iheya_n/NBC/{}/{}_{}".format(species[2], species[1], boxsize)
#        readtxt = open(path + "/{}{}_{}.txt".format(species[1], 'test', boxsize), "r")
#        string = string + readtxt.read()

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
    ImgArrayTrain = open(path + "/{}train_{}.txt".format(species[1], boxsize), "a")
    ImgArrayTest = open(path + "/{}test_{}.txt".format(species[1], boxsize), "a")
    for i in trainingset:
        Io = Image.open(path + "/" + imgname + "_{}_all/{}-{}.tif".\
                        format(boxsize, imgname, "%05d" % i)) #"%05d" % n)
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
                        format(boxsize, imgname, "%05d" % i)) #"%05d" % n)
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
        readtxt = open(path + "/{}{}_{}.txt".format(species[1], i, boxsize), "r")
        string = readtxt.read()
        stringtolist = [int(s) for s in string[:-1].split(",")]
        buffer = bytes(stringtolist)
        strfile = r"{}/{}{}_{}.bin".format(path, species[1], i, boxsize)
        with open(strfile, 'bw') as f:
            f.write(buffer)

def txttobinconvertertestcombine():
    selectedspecies = [BJa, BPl, SCr]
    string = ""
    for i in selectedspecies:
        species = i
        path = "D:/Iheya_n/NBC/{}/{}_{}".format(species[2], species[1], boxsize)
        readtxt = open(path + "/{}{}_{}.txt".format(species[1], 'test', boxsize), "r")
        string = string + readtxt.read()
    stringtolist = [int(s) for s in string[:-1].split(",")]
    buffer = bytes(stringtolist)
    strfile = r"{}/{}{}_{}.bin".format(path[:10], "BJaBPlSCr", 'test', boxsize)
    with open(strfile, 'bw') as f:
        f.write(buffer)

#mosaic_crop(excelfile, mosfile, boxsize, imgname)
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