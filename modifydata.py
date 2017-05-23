"""
This is a program to modify our data extract file from IPUMS.
The data extract input file is a list of integers strings.
This program is used to splice each string correctly according
to our features text, remove patients in which CNBRES(whether
they have breast cancer or not) is not yes/no, and "move"
CNBRES to the last feature in the string. The program
outputs a csv file with the modified data.
Written by: Maggie Eberts and Tiffany Yu
"""
import random

def main():

  # read in DataFeatures file, store in variable and varlen
  f1 = open("projectData/DataFeatures.txt", 'r')
  i=0
  variable=[]
  varlen=[]
  for line in f1:
    if (i<6):
      i +=1
      continue
    if (i>104):
      break
    else:
      info = line.split()
      variable.append(info[0])
      varlen.append(int(info[3]))
      i+=1

  #open original data file
  f2 = open("projectData/Cancer.dat", 'r')
  data = []
  dataNo = []
  for line in f2:
    line =line.strip()
    instance = [] #one example's feature vector
    index = 0 #where we are in the line
    for i in range(len(variable)-1): #for each variable(feature)
      instance.append(int(line[index:index+varlen[i]]))
      index+= varlen[i]
    if (instance[57]==0 or instance[57]==1 or instance[57]==2): #0or1=no/2=yes, only add these to data
      inst = []
      inst = instance[:57] + instance[58:]
      if(instance[57]==2): #CNBRES = true
        inst.append(1)
        data.append(inst)
      else:
        inst.append(0) #NIU or no = false
        dataNo.append(inst)
  #add 439 negative examples so that our data set is 50:50 for 0:1 classification ratio
  indexList = random.sample(xrange(89520), 439)
  for i in indexList:
    data.append(dataNo[i])
  random.shuffle(data)

  #write modified data to new file to be used for experiments
  f3 = open("projectData/ModCancerData.csv", 'w')
  for line in data:
    for feature in line:
      f3.write("%s " %feature)
    f3.write("\n")

  #create a smaller debugging data set
  f4 = open("projectData/debug.csv",'w')
  count = 0
  for line in data:
    if count%100 == 0:
        for feature in line:
            f4.write("%s " %feature)
        f4.write("\n")
    count += 1

main()
