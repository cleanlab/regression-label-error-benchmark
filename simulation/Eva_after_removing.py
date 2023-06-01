import utils
import pickle
import numpy as np


run=50
n_boot=20
Signal_strength=[-3,-2,-1,0,1,2,3]
Res_dict=dict()
Sample_size=[200]
for n in Sample_size:
    for s in Signal_strength:
        for l in range(run):
            AU_dic={};Out_prop={};Removed_prop={}
            Data=pickle.load(open('./Data/Data5DLM_'+str(n)+"_"+str(0.1)+'/Data'+str(s) +'.pkl' ,'rb'))
            Scores=utils.Eva_scores(Data,l)
            remove_list=utils.removed_list(Data,l,Scores)
            for s_type in ['res','ari','geo']:
                Au_mat=np.zeros([2,3])
                S_temp,Removed_prop[s_type],Out_prop[s_type]=utils.Eva_a_score(Data,l,remove_list[s_type])
                Au_mat[0,0],Au_mat[1,0]=utils.Get_au(Data,l,S_temp['res'],"train" )
                Au_mat[0,1],Au_mat[1,1]=utils.Get_au(Data,l,S_temp['ari'],"train" )
                Au_mat[0,2],Au_mat[1,2]=utils.Get_au(Data,l,S_temp['geo'],"train" )
                AU_dic[s_type]=Au_mat
            Res_dict["5DLM"+"_"+str(n)+'_'+str(0.1)+'_'+str(s)+'_'+str(l) ]={'AU':AU_dic,'Out_prop':Out_prop,'Removed_prop':Removed_prop}
pickle.dump(Res_dict, open('5DLM_MSE_GBM_AG'+str(n) + '.pkl', 'wb' ))   