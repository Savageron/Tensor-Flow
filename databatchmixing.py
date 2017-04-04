import random

recordbytelength = 50*50*3+1

def databatchmix15Train(data_path,numberoftrainingsamples):
        species1 = []
        species2 = []
        species3 = []
        for i in ["1","2","3"]:
                data = open(data_path+"data_batch_{}.txt".format(i),"r")
                string = data.read()
                stringtolist = [int(s) for s in string[:-1].split(",")]
                if i == "1":
                        species1 = species1 + stringtolist
                elif i == "2":
                        species2 = species2 + stringtolist
                elif i == "3":
                        species3 = species3 + stringtolist
        allspecies = species1 + species2 + species3
        data_batch_1_list = []
        data_batch_2_list = []
        data_batch_3_list = []
        k = 0
        for i in random.sample(range(numberoftrainingsamples), numberoftrainingsamples):
                if k<numberoftrainingsamples/3:
                        data_batch_1_list = data_batch_1_list + allspecies[int(recordbytelength*i):int(recordbytelength+recordbytelength*i)]
                        k=k+1
                elif k<numberoftrainingsamples*2/3:
                        data_batch_2_list = data_batch_2_list + allspecies[int(recordbytelength*i):int(recordbytelength+recordbytelength*i)]
                        k=k+1
                else:
                        data_batch_3_list = data_batch_3_list + allspecies[int(recordbytelength*i):int(recordbytelength+recordbytelength*i)]
                        k=k+1
        for i in [[data_batch_1_list,"1"], [data_batch_2_list,"2"], [data_batch_3_list,"3"]]:
                new_data_batch_string = ','.join(str(e) for e in i[0]) + ','
                new_data_batch = open(data_path + "data_batch_{}_new.txt".format(i[1]),"a")
                new_data_batch.write(new_data_batch_string)

#data_path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/NewResults/3to1ratio/15Train/"
data_path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/NewResults/3to1ratio/15750Train/"
#databatchmix15Train(data_path,15)
databatchmix15Train(data_path,15750)

#for i in [["4to1ratio",4200],["2to1ratio",6999],["1to1ratio",10500],["1to2ratio",13998],["1to3ratio",15750],["1to4ratio",16800]]:
#        data_path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/NewResults/{}/".format(i[0])
#        ratiochangefix(data_path,i[1])
