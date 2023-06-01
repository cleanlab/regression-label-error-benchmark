import utils
import numpy as np
import pickle


##conformal
run=50
n_boot=20
Out_ratio_train=0
Signal_strength=[-3,-2,-1,0,1,2,3]
Res_dict={}
for s in range(7):
    Data=pickle.load(open('Data/Data5D_200_'+str(Out_ratio_train) +'/Data'+ str(s)+'.pkl', 'rb'))
    Res_met={'ari':np.zeros([run,2]),'res':np.zeros([run,2]),'geo':np.zeros([run,2])}
    for l in (range(run)): 
        trained_model=utils.Conformal_train_atg2(Data['train'],l,n_boot=20,run=50 )
        con_res=utils.Conformal_eva_atg2(Data,trained_model,l)
        Res_met['res'][l,0]=con_res['res']['FDR'];Res_met['res'][l,1]=con_res['res']['Power']
        Res_met['ari'][l,0]=con_res['ari']['FDR'];Res_met['ari'][l,1]=con_res['ari']['Power']
        Res_met['geo'][l,0]=con_res['geo']['FDR'];Res_met['geo'][l,1]=con_res['geo']['Power']
    Res_dict[str(Signal_strength[s])]=Res_met
pickle.dump(Res_dict, open('5D_'+ str(Out_ratio_train)+'.pkl' ,'wb' ))



