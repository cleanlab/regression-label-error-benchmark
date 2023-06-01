import utils
import numpy as np
import pickle
from joblib import Parallel, delayed


Sig_level=[-3,-2,-1,0,1,2,3]

for Out_ratio_train in [0,0.1]:
    def Get_result(s):
        run=50
        n_boot=20
        Data=pickle.load(open('Data/Data5DLM_200_'+str(Out_ratio_train) +'/Data'+ str(s)+'.pkl', 'rb'))
        Res_met={'ari':np.zeros([run,2]),'res':np.zeros([run,2]),'geo':np.zeros([run,2])}
        for l in (range(run)):  #each training
            trained_model=utils.Conformal_train(Data['train'],l,n_boot=20,run=50 )
            con_res=utils.Conformal_eva(Data,trained_model,l)
            Res_met['res'][l,0]=con_res['res']['FDR'];Res_met['res'][l,1]=con_res['res']['Power']
            Res_met['ari'][l,0]=con_res['ari']['FDR'];Res_met['ari'][l,1]=con_res['ari']['Power']
            Res_met['geo'][l,0]=con_res['geo']['FDR'];Res_met['geo'][l,1]=con_res['geo']['Power']
        return Res_met
    Sig_level=[-3,-2,-1,0,1,2,3]
    Results = Parallel(n_jobs=8)(delayed(Get_result)(s) for s in Sig_level)
    pickle.dump(Results, open("Res_mat_"+str(Out_ratio_train)+".pkl","wb" ) )

