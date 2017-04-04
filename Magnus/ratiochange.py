import random
import sys
pathof3to1ratiototal = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/3to1ratio/15750Train/num_iter_10000_5250Test"
#"D:/Iheya_n/HvassTutResults/1BJaBPlSCr/3to1ratio/21000pax"

import cifar10

recordbytelength = 50*50*3+1

def ratiochange(traintotestratio,traintotestratiofoldername):
        numberoftrainingsamples = 5250*4*traintotestratio/(traintotestratio+1)
        while numberoftrainingsamples % 3 != 0:
               numberoftrainingsamples = numberoftrainingsamples + 1
        else:
             numberoftestsamples = 5250*4 - numberoftrainingsamples
        print ("train to test ratio is {}, folder is {}".format(traintotestratio, traintotestratiofoldername))
        print ("number of training samples is {}".format(numberoftrainingsamples))
        print ("number of test samples is {}".format(numberoftestsamples))
        filenames = ["data_batch_1","data_batch_2","data_batch_3","test_batch"]
        BJa = []
        BPl = []
        SCr = []
        for i in filenames:
                data = open(pathof3to1ratiototal+"/"+"{}.txt".format(i),"r")
                string = data.read()
                stringtolist = [int(s) for s in string[:-1].split(",")]
                if i == "data_batch_1":
                        BJa = BJa + stringtolist
                elif i == "data_batch_2":
                        BPl = BPl + stringtolist
                elif i == "data_batch_3":
                        SCr = SCr + stringtolist
                else:
                        BJa = BJa + stringtolist[0:int(recordbytelength*5250/3)]
                        BPl = BPl + stringtolist[int(recordbytelength*5250/3):int(recordbytelength*5250*2/3)]
                        SCr = SCr + stringtolist[int(recordbytelength*5250*2/3):int(recordbytelength*5250)]
        if len(BJa) == len(BPl) == len(SCr) == 52507000:
                print ("length of each BJa BPl SCr stringtolist == (50*50*3+1)*5250*4/3 == 7000pics*7501bytes== 52507000")
        else:
                sys.exit("aa!errors!")
        data_batch_1_list = []
        data_batch_2_list = []
        data_batch_3_list = []
        test_batch_list = []
        for i in [[BJa,data_batch_1_list,"data_batch_1.txt"],[BPl,data_batch_2_list,"data_batch_2.txt"],[SCr,data_batch_3_list,"data_batch_3.txt"]]:
                k = 0
                for j in random.sample(range(7000), 7000):
                        if k < numberoftrainingsamples/3:
                                i[1] = i[1] + i[0][int(recordbytelength*j):int(recordbytelength+recordbytelength*j)]
                                k=k+1
                                #if k/500==int:
                                if k%1000==0:
                                        print ("k={}".format(k))
                                        print (len(i[1]))
                        else:
                                test_batch_list = test_batch_list + i[0][int(recordbytelength*j):int(recordbytelength+recordbytelength*j)]
                                k=k+1
                                if k%1000==0:
                                        print ("k={}".format(k))
                                        print (len(i[1]))
                print ("Check if length of data_batch list is {} == {}".format(len(i[1]),recordbytelength*numberoftrainingsamples/3))
                print ("Check if length of test_batch list is {} == {}".format(len(test_batch_list),recordbytelength*numberoftestsamples/3))
                print ("==========================")
                path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/{}/".format(traintotestratiofoldername)
                print (i[1][:10])
                print (','.join(str(e) for e in i[1][:20]) + ',')
                data_batch_string = ','.join(str(e) for e in i[1]) + ','
                data_batch = open(path + i[2], "a")
                data_batch.write(data_batch_string)
        test_batch_string = ','.join(str(e) for e in test_batch_list) + ','
        test_batch = open(path + "test_batch.txt", "a")
        test_batch.write(test_batch_string)
        #print ("Check if length of data_batch_1_list is {} == {}".format(len(data_batch_1_list),recordbytelength*numberoftrainingsamples/3))
        #print ("Check if length of data_batch_2_list is {} == {}".format(len(data_batch_2_list),recordbytelength*numberoftrainingsamples/3))
        #print ("Check if length of data_batch_3_list is {} == {}".format(len(data_batch_3_list),recordbytelength*numberoftrainingsamples/3))
        #print ("Check if length of test_batch_list is {} == {}".format(len(test_batch_list),recordbytelength*numberoftestsamples))
        #print ("==========================")
        #path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/{}/".format(traintotestratiofoldername)
        #data_batch_1_string = ','.join(str(e) for e in data_batch_1_list) + ','
        #data_batch_1 = open(path + "data_batch_1.txt", "a")
        #data_batch_1.write(data_batch_1_string)
        #data_batch_2_string = ','.join(str(e) for e in data_batch_2_list) + ','
        #data_batch_2 = open(path + "data_batch_2.txt", "a")
        #data_batch_2.write(data_batch_2_string)
        #data_batch_3_string = ','.join(str(e) for e in data_batch_3_list) + ','
        #data_batch_3 = open(path + "data_batch_3.txt", "a")
        #data_batch_3.write(data_batch_3_string)
        #test_batch_string = ','.join(str(e) for e in test_batch_list) + ','
        #test_batch = open(path + "test_batch.txt", "a")
        #test_batch.write(test_batch_string)
#        for i in random.sample(range(len(5250*4)),numberoftrainingsamples):

#        for i in filenames[:3]:
#                print (i)
#        i = filenames[3]
#        print (i)
#        sys.exit("123")
#        filecheck = f.open(pathof3to1ratiototal)

differentratios = [[4/1,"4to1ratio"],[2/1,"2to1ratio"],[1/1,"1to1ratio"],[1/2,"1to2ratio"],[1/3,"1to3ratio"],[1/4,"1to4ratio"]]
for i in differentratios:
        ratiochange(i[0],i[1])



#5250 images in each 4 files. total 5250*4. training 5250*3, test 5250*1
def pax2100():
        pax = 1575
        filenames = ["data_batch_1","data_batch_2","data_batch_3","test_batch"]
        for i in filenames:
                data = open(pathof3to1ratiototal+"/"+"{}.txt".format(i),"r")
                string = data.read()
                stringtolist = [int(s) for s in string[:-1].split(",")]
                emptylist = []
                for j in random.sample(range(5250), 525):
                        emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
                newstring = ",".join(str(e) for e in emptylist)
                f = open(pathof3to1ratiototal[:-13]+"{}Train/{}.txt".format(pax,i),"a")
                f.write(newstring+",")

def pax210():
        #52 samples in each train_batch_, 54 samples in test_batch
        pax = 156
        filenames = ["data_batch_1","data_batch_2","data_batch_3"]
        for i in filenames:
                data = open(pathof3to1ratiototal+"/"+"{}.txt".format(i),"r")
                string = data.read()
                stringtolist = [int(s) for s in string[:-1].split(",")]
                emptylist = []
                for j in random.sample(range(5250), 52):
                        emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
                newstring = ",".join(str(e) for e in emptylist)
                f = open(pathof3to1ratiototal[:-13]+"{}Train/{}.txt".format(pax,i),"a")
                f.write(newstring+",")
        testfilename = "test_batch"
        data = open(pathof3to1ratiototal+"/"+"{}.txt".format(testfilename),"r")
        string = data.read()
        stringtolist = [int(s) for s in string[:-1].split(",")]
        emptylist = []
        for j in random.sample(range(5250), 54):
                emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
        newstring = ",".join(str(e) for e in emptylist)
        f = open(pathof3to1ratiototal[:-13]+"{}Train/{}.txt".format(pax,testfilename),"a")
        f.write(newstring+",")

def pax21():
        #5 samples in each train_batch_, 6 samples in test_batch
        pax = 15
        filenames = ["data_batch_1","data_batch_2","data_batch_3"]
        for i in filenames:
                data = open(pathof3to1ratiototal+"/"+"{}.txt".format(i),"r")
                string = data.read()
                stringtolist = [int(s) for s in string[:-1].split(",")]
                emptylist = []
                for j in random.sample(range(5250), 5):
                        emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
                newstring = ",".join(str(e) for e in emptylist)
                f = open(pathof3to1ratiototal[:-13]+"{}Train/{}.txt".format(pax,i),"a")
                f.write(newstring+",")
        testfilename = "test_batch"
        data = open(pathof3to1ratiototal+"/"+"{}.txt".format(testfilename),"r")
        string = data.read()
        stringtolist = [int(s) for s in string[:-1].split(",")]
        emptylist = []
        for j in random.sample(range(5250), 6):
                emptylist = emptylist + stringtolist[0+recordlengthbyte*j:recordlengthbyte+recordlengthbyte*j]
        newstring = ",".join(str(e) for e in emptylist)
        f = open(pathof3to1ratiototal[:-13]+"{}Train/{}.txt".format(pax,testfilename),"a")
        f.write(newstring+",")

#pax2100()
#pax210()
#pax21()
