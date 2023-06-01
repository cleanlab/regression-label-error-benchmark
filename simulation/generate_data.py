import numpy as np
import pickle
import utils
import os

n=200
Sig_level=[-3,-2,-1,0,1,2,3]
Out_ratio=0

if not os.path.isdir("./Data/Data5D_"+str(n)+"_"+str(Out_ratio)):
    os.mkdir("./Data/Data5D_"+str(n)+"_"+str(Out_ratio))


for s in Sig_level:
    Data=utils.Get_data5D(n,n,run=50,outlier=[Out_ratio,0.1],signal=s)
    pickle.dump(Data, open( "./Data/Data5D_"+str(n)+"_"+str(Out_ratio)+'/Data'+ str(s)+'.pkl' ,"wb"))

#### 5-D linear model
import numpy as np
import pickle
import utils
import os

n=1000
Sig_level=[-3,-2,-1,0,1,2,3]
Out_ratio=0

if not os.path.isdir("./Data/Data5DLM_"+str(n)+"_"+str(Out_ratio)):
    os.mkdir("./Data/Data5DLM_"+str(n)+"_"+str(Out_ratio))


for s in Sig_level:
    Data=utils.Get_data5DLM(n,n,run=50,outlier=[Out_ratio,0.1],signal=s)
    pickle.dump(Data, open( "./Data/Data5DLM_"+str(n)+"_"+str(Out_ratio)+'/Data'+ str(s)+'.pkl' ,"wb"))




