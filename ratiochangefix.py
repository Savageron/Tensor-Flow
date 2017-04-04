def ratiochangefix(data_path,numberoftestsamples):
        data = open(data_path+"test_batch.txt","r")
        string = data.read()
        stringtolist = [int(s) for s in string[:-1].split(",")]
        numberofbytesdesired = (50*50*3+1)*numberoftestsamples
        desiredbytes = stringtolist[-numberofbytesdesired:]
        print ("check if length of desiredbytes {} == {}".format(len(desiredbytes),numberofbytesdesired))
        new_test_batch_string = ','.join(str(e) for e in desiredbytes) + ','
        new_test_batch = open(data_path + "test_batch_new.txt","a")
        new_test_batch.write(new_test_batch_string)

for i in [["4to1ratio",4200],["2to1ratio",6999],["1to1ratio",10500],["1to2ratio",13998],["1to3ratio",15750],["1to4ratio",16800]]:
        data_path = "D:/Iheya_n/HvassTutResults/2BatSCrNSe/NewResults/{}/".format(i[0])
        ratiochangefix(data_path,i[1])
