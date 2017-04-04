import random
import sys
path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/pax/21000pax"

recordlengthbyte = 50*50*3+1

#5250 images in each 4 files. total 5250*4. training 5250*3, test 5250*1
def pax2100():
        pax = 2100
        filenames = ["data_batch_1","data_batch_2","data_batch_3","test_batch"]
        for i in filenames:
                data = open(path+"/"+"{}.txt".format(i),"r")
                string = data.read()
                stringtolist = [int(s) for s in string[:-1].split(",")]
                emptylist = []
                for j in random.sample(range(5250), 525):
                        emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
                newstring = ",".join(str(e) for e in emptylist)
                f = open(path[:-8]+"{}pax/{}.txt".format(pax,i),"a")
                f.write(newstring+",")

def pax210():
        #52 samples in each train_batch_, 54 samples in test_batch
        pax = 210
        filenames = ["data_batch_1","data_batch_2","data_batch_3"]
        for i in filenames:
                data = open(path+"/"+"{}.txt".format(i),"r")
                string = data.read()
                stringtolist = [int(s) for s in string[:-1].split(",")]
                emptylist = []
                for j in random.sample(range(5250), 52):
                        emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
                newstring = ",".join(str(e) for e in emptylist)
                f = open(path[:-8]+"{}pax/{}.txt".format(pax,i),"a")
                f.write(newstring+",")
        testfilename = "test_batch"
        data = open(path+"/"+"{}.txt".format(testfilename),"r")
        string = data.read()
        stringtolist = [int(s) for s in string[:-1].split(",")]
        emptylist = []
        for j in random.sample(range(5250), 54):
                emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
        newstring = ",".join(str(e) for e in emptylist)
        f = open(path[:-8]+"{}pax/{}.txt".format(pax,testfilename),"a")
        f.write(newstring+",")

def pax21():
        #5 samples in each train_batch_, 6 samples in test_batch
        pax = 21
        filenames = ["data_batch_1","data_batch_2","data_batch_3"]
        for i in filenames:
                data = open(path+"/"+"{}.txt".format(i),"r")
                string = data.read()
                stringtolist = [int(s) for s in string[:-1].split(",")]
                emptylist = []
                for j in random.sample(range(5250), 5):
                        emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
                newstring = ",".join(str(e) for e in emptylist)
                f = open(path[:-8]+"{}pax/{}.txt".format(pax,i),"a")
                f.write(newstring+",")
        testfilename = "test_batch"
        data = open(path+"/"+"{}.txt".format(testfilename),"r")
        string = data.read()
        stringtolist = [int(s) for s in string[:-1].split(",")]
        emptylist = []
        for j in random.sample(range(5250), 6):
                emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
        newstring = ",".join(str(e) for e in emptylist)
        f = open(path[:-8]+"{}pax/{}.txt".format(pax,testfilename),"a")
        f.write(newstring+",")
