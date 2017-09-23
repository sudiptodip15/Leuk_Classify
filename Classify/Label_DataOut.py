''' Extracts label from patient data using data cleaning '''

fl=open('../Data/GSE13159.info.txt','r')
fl.readline
patient={}
while(True):
    line=fl.readline().rstrip()
    if(len(line)==0):
        break
    line=line.split('\t')
    patient[line[0]]=line[2]


fD=open('../Data/mile_cleaned.csv','r')
line=fD.readline().rstrip().split(',')
fnewl=open('../Data/Label.txt','w')
for i in range(1,len(line)):
    key=line[i][1:-1]
    if(patient.get(key,0)==0):
        continue
    else:
        fnewl.write(key+'\t'+patient[key]+'\n')


fl.close()
fD.close()
fnewl.close()

