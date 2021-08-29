# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:55:06 2021

@author: jeanl
"""

##Predict function for bnlearn

def bnpredict(model, df, bn_dict, var):
    #check if correct package is loaded
    import bnlearn as bn
        
    #check for variables that need to be dropped
    complist1 = list(bn_dict["pos"].keys())
    complist2 = list(df.columns)
    complist3 = list(set(complist1).symmetric_difference(set(complist2)))
    df = df.drop(complist3, axis='columns')
    
    pred_l1 = []
    pred_l1_prob = []
    pred_l1_prob_neq = []
    counter1 = 0
    for index, row in df.iterrows():
        temp = dict(row)
        temp.pop(var)
        temp2 = bn.inference.fit(model, variables=[var], 
                     evidence=temp, verbose = 0)
        #pdays not in this model
        temp3 = temp2.df
        temp4 = temp3.groupby("y")["p"].max().sort_values(ascending = False).reset_index()
        temp5a = temp4["y"].iloc[0]
        temp5b = temp4["p"].iloc[0]
        temp5c = temp4["p"].iloc[1]
        pred_l1.append(temp5a)
        pred_l1_prob.append(temp5b)
        pred_l1_prob_neq.append(temp5c)
        counter1+=1
        print(counter1)
    
    return pred_l1, pred_l1_prob, pred_l1_prob_neq
