#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:50:55 2018
Attempt to do a golf example
@author: edmond
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import isclose



def RewardFunction(s,a,s_g,N):
    if s == s_g:
        return(0)
    elif s!=s_g and a == 1:
        return(-1)
    elif s!=s_g and a == 2:       
        return(-4)
    elif s!=s_g and a ==3:
        return(-N/4)
        

        
def reward_vector(N,a,s_g):
    r=[]
    for si in range(N):
        value=RewardFunction(si,a,s_g,N)
        r.append(value)
    r=np.asarray(r).reshape(N,1)
    return(r)


       
        
#### transition probability matrices #######
def make_trans_prob(long,embrace_States,prob,N,s_g):
    matriz=[]
    aux=np.ravel(np.zeros(shape=(1,N)))
    for i in range(N):
        inicio=i+long
        aux=np.ravel(np.zeros(shape=(1,N)))
        inicio,fin=int((inicio-np.round(embrace_States/2))%N),int((inicio+np.round(embrace_States/2))%N)
        if i == s_g:
            inicio=inicio-1
        if inicio <  s_g and fin > inicio:
            aux[inicio:fin]=prob
            if len(aux[inicio:fin])<embrace_States:
                aux[inicio:fin+1]=prob
                
        elif inicio >= s_g and fin > inicio:
            inicio=inicio
            aux[inicio:fin]=prob
            if len(aux[inicio:fin])<embrace_States:
                aux[inicio:fin+1]=prob

        elif inicio >= s_g and fin < inicio:

            if len(aux[inicio:])< embrace_States:
#                print('aqui',inicio)

                fin=embrace_States-len(aux[inicio:])
                if  fin < embrace_States:
                    aux[inicio:]=prob
                    aux[:fin]=prob
            elif len(aux[inicio:])== embrace_States:
                    aux[inicio:]=prob
            if inicio >= N:
                aux[:fin]=prob
#
        elif inicio == s_g:
            pass
        else:
            print(inicio,s_g,fin)
            raise ValueError (' condicion no condisdera')
        matriz.append(aux)
    
    matriz=np.asarray(matriz)
#    print(np.sum(matriz,axis=1))
    return(matriz)
 
def inv(X):
    Xhat=np.linalg.pinv(X)
    return(Xhat)    
def create_random_policies(N):
    policies_random=[]
    for i in range(N):
        array=np.ravel(np.zeros(shape=(1,3)))
        ix_ran=int(np.random.randint(3))
        array[ix_ran]=1
        policies_random.append(array)
    policies_random=np.asarray(policies_random)
    return(policies_random)
    
Number_of_States=100
s_g=50

N=Number_of_States
#### first action and P#####
a1=1
embrace_1=2
prob_1=1/2
long_1=2
#long_1=0
P1=make_trans_prob(long_1,embrace_1,prob_1,N,s_g); P1[s_g,:]=0;# P1[s_g,s_g]=1
R1=reward_vector(N,1,s_g)
### second action and P#####
a2=2
embrace_2=4
prob_2=1/4
long_2=6
#long_2=0

P2=make_trans_prob(long_2,embrace_2,prob_2,N,s_g); P2[s_g,:]=0; #P2[s_g,s_g]=1
R2=reward_vector(N,2,s_g)

##
##### third action and P#####
a3=3
embrace_3=5
prob_3=1/5
long_3=np.round(N/2)-2
#long_3=0
R3=reward_vector(N,3,s_g)
P3=make_trans_prob(long_3,embrace_3,prob_3,N,s_g);P3[s_g,:]=0; #P3[s_g,s_g]=1

###############three diferents policies for all states #### NO ARBITRARY ######+

##### 3 policy: for action 1,2,3  for all states ######
P=np.array([P1,P2,P3])
R=np.array([R1,R2,R3])
policy=np.zeros(shape=(100,3)); policy[:,0]=1
#plt.close('all')
#plt.figure
##
#V=iterative_policy_Evaluation(P,R,s_g,N,policy,theta=0.0001)
#plt.plot(range(N),V)
##v_a_1=inv((np.eye(N)-P1))@R1

##v_a_3=inv((np.eye(N)-P3))@R3
##plt.plot(range(N),v_a_1,c=(0,0,0),label='action1')
###plt.plot(range(N),v_a_2,c=(0,0.5,0),label='acion2')
###plt.plot(range(N),v_a_3,c=(0,1,0),label='action3')
##plt.ylabel('Value functions of three policyes for all states')
##plt.xlabel('States')
###plt.figure()
###### 3 determinsitic policy for all states#### arbitrart#######
###plt.close('all')
##P=np.array([P1,P2,P3])
##R=np.array([R1,R2,R3])
#policies_random=create_random_policies(N)
#V=iterative_policy_Evaluation(P,R,s_g,N,policies_random,theta=0.0001)
#plt.plot(range(N),V,c=(1,1,0),label='action random')

#################policy iteration ###########
#init_i=(V_random,policies_random,policy_stable,S)        


def policy_evaluation(P,R,init,theta=0.0001):
   

    V_,policy_,S=init
    V=V_.copy()
    policy=policy_.copy()
    i=0
    policy_stable=False
    while policy_stable==False:
        delta=0
        for s,policy_ in enumerate(policy):
            v_=V[s].copy()
#            print('cosa',np.where(policy_ !=0)[0])
            a=int(np.where(policy_ !=0)[0])
            V[s]=float(R[a,s,:]+P[a,s,:]@V)
            diff=np.abs(v_-V[s])
            delta=max([delta,diff])
            if delta < theta :
                policy_stable=True
        i=i+1
#        print('V',np.sum(V),'\ndiff',diff)
    V=np.asarray(V)
    return(V,policy,S)


def policy_iteration4(P,R,init,theta=0.0001):

    V_,pi_,S=policy_evaluation(P,R,init)
    V=V_.copy()
    pi=pi_.copy()
    
    
    policy_stable=True
    number_of_actions=3

    if policy_stable==True:
        for s in range(S):
            old_pi=pi[s].copy()
            old_action=np.where(old_pi!=0)[0]
            V_i=[]
            for a in range(number_of_actions):
                aux=float(R[a,s,:]+P[a,s,:]@V)
                V_i.append(aux)
            new_action=np.argmax(V_i)
            new_pi=np.zeros(shape=3)
            new_pi[new_action]=1
            V_new_s=np.max(V_i)
            if new_action != old_action  or V_new_s <V[s]:
                policy_stable = False
                while policy_stable==False:

                    init=(V,pi,S)
                    V,pi,S=policy_evaluation(P,R,init)
                    
                    old_pi=pi[s].copy()
                    old_action=np.where(old_pi!=0)[0]
                    V_i=[]
                    for a in range(number_of_actions):
                        aux=float(R[a,s,:]+P[a,s,:]@V)
                        V_i.append(aux)
                    new_action=np.argmax(V_i)
                    new_pi=np.zeros(shape=3)
                    new_pi[new_action]=1
                    V_new_s=np.max(V_i)
                    if (new_action != old_action) or V_new_s <V[s] :
                        policy_stable=False
                        pi[s]=new_pi
                        init=(V,pi,S)
                        V,pi,S=policy_evaluation(P,R,init)


                    else:
                        policy_stable=True

            else:
                policy_stable=True

        return(V,pi)
                        
                    
                
    

    
    
def policy_iteration(P,R,init,theta):

    V_,pi_,S=policy_evaluation(P,R,init)
    number_of_actions=3
    V=V_.copy()
    pi=pi_.copy()

    policy_stable=True
    
    if policy_stable == True:
    
    
        for s in range(S):
            old_pi=pi[s].copy()
            old_action=np.where(old_pi != 0)[0]
            aux_array=[]
            for a in range(number_of_actions):
                aux=float(R[a,s,:]+P[a,s,:]@V)
                aux_array.append(aux)
            new_action=np.argmax(aux_array)
            new_pi=np.zeros(shape=3)
            new_pi[new_action]=1
            new_V=np.max(aux_array)
    #        V[s]=np.max(aux_array)
            if old_action != new_action or new_V < V[s]:
                policy_stable=False
                i=0
                while policy_stable==False:
                    init=(V,pi,S)
                    V,pi,S=policy_evaluation(P,R,init)
                    old_pi=pi[s].copy()
                    old_action=np.where(old_pi != 0)[0]
                    aux_array=[]
                    for a in range(number_of_actions):
                        aux=float(R[a,s,:]+P[a,s,:]@V)
                        aux_array.append(aux)
                    new_action=np.argmax(aux_array)
                    new_pi=np.zeros(shape=3)
                    new_pi[new_action]=1
                    new_V=np.max(aux_array)
    #                V[s]=np.max(aux_array)
                    if old_action != new_action or new_V < V[s]:
                        policy_stable=False
                        pi[s]=new_pi
                        init=(V,pi,S)
                        V,pi,S=policy_evaluation(P,R,init)
                        i=i+1
                    else:
                        policy_stable=True
 
    
                    
    
            else:
                policy_stable=True
         

                        
        
            
#        return(np.asarray(V_array),np.asarray(pi_array))
        return(V,pi)
            
    


                
            

def policy_interation2(P,R,init,theta=0.00001):
    
    
    V_,pi_,S=policy_evaluation(P,R,init)
    Number_of_possible_action=3
    V=V_.copy()
    pi=pi_.copy()
    policy_stable=True

    
    if policy_stable==True:
    
        
        for s in range(S):
            old_action=pi[s].copy()
            aux_array=[]
            for a in range(Number_of_possible_action):
                aux=float(R[a,s,:]+P[a,s,:]@V)
                aux_array.append(aux)
            new_action=np.argmax(aux_array)
            new_pi=np.zeros(shape=3)
            new_pi[new_action]=1
            new_V_s=np.max(aux_array)
            if np.any(old_action != new_pi) or new_V_s < V[s]:
                policy_stable=False
                i=0
                while policy_stable==False:
                    
#                    V[s]=new_V_s
#                    pi[s]=new_pi
                    init=(V,pi,S)
                    V,pi,S=policy_evaluation(P,R,init)
                    
                    old_action=pi[s].copy()
                    aux_array=[]
                    for a in range(Number_of_possible_action):
                        aux=float(R[a,s,:]+P[a,s,:]@V)
                        aux_array.append(aux)
                    new_action=np.argmax(aux_array)
                    new_pi=np.zeros(shape=3)
                    new_pi[new_action]=1
                    new_V_s=np.max(aux_array)
                    if np.any(old_action != new_pi) or new_V_s < V[s]:
                        policy_stable=False
                        pi[s]=new_pi
                        init=(V,pi,S)
                        V,pi,S=policy_evaluation(P,R,init)
                        i=i+1
                    
                    else:
                        policy_stable=True


            else:
                policy_stable=True

                
                        
        return(V,pi)

#17 cifras
def value_iteration(P,R,S,init,s_g=50,theta=0.0001):
    V_,pi_,S=init
    V=V_.copy()
    pi=pi_.copy()
    V[s_g]=0
    number_of_actions=3
    i=0
    pi=np.zeros(shape=(S,3))
    while True:
        delta=0
        for s in range(S):
            v=V[s].copy()
            aux_array=[]
            for a in range(number_of_actions):
                aux=float(R[a,s,:]+P[a,s,:]@V)
                aux_array.append(aux)
            V[s]=np.max(aux_array)
            diff=np.abs(v-V[s])
            delta=np.max([delta,diff])
            pi_s=np.zeros(shape=3)
            pi_s[np.argmax(aux_array)]=1
            pi[s,:]=pi_s
            if delta < theta:

#                print(i)
                return(V,pi)
                break
            i=i+1
    
            
def policy_iteration3(P,R,init,theta=0.0001):
    
    
    V_,pi_,S=policy_evaluation(P,R,init)
    V=V_.copy()
    pi=pi_.copy()
    Number_of_possible_action=3

    policy_stable =True
    if policy_stable==True:
        for s in range(S):
                policy_stable=False
                i=0
                while policy_stable==False:
                    init=(V,pi,S)
                    V,pi,S=policy_evaluation(P,R,init)
                    old_action=pi[s].copy()
                    aux_array=[]
                    for a in range(Number_of_possible_action):
                        aux=float(R[a,s,:]+P[a,s,:]@V)
                        aux_array.append(aux)
                    new_action=np.argmax(aux_array)
                    new_pi=np.zeros(shape=3)
                    new_pi[new_action]=1
                    V_new_s=np.max(aux_array)
                    if np.any(old_action != new_pi) or V_new_s < V[s]:
                        policy_stable=False  
                        pi[s,:]=new_pi
                        init=(V,pi,S)
                        V,pi,S=policy_evaluation(P,R,init)
                        i=i+1

                    else:
                        policy_stable=True
                    


                    
        return(V,pi)
        
     
def iterative_policy_evaluation(P,R,init,theta=0.0001):
   
    Number_of_possible_action=3

    V_,policy_,S=init
    V=V_.copy()
    policy=policy_.copy()
    i=0
    policy_stable=False
    while policy_stable==False:
        delta=0
        for s,policy_ in enumerate(policy):
            v_=V[s].copy()
            
            aux_array=[]
            for a in range(Number_of_possible_action):
                aux=float(R[a,s,:]+P[a,s,:]@V)
                aux_array.append(aux)
            aux_array=np.asarray(aux_array)
            V[s]=np.sum(aux_array)
            diff=np.abs(v_-V[s])
#            print(diff,diff==0)

            delta=max([delta,diff])
            if  diff==0:
                policy_stable=True
        i=i+1
#        print('V',np.sum(V),'\ndiff',diff)
    V=np.asarray(V)
    return(V,policy,S)
        
        
    
def random_init_q(random,S=100):
    if random==True:
        V_random_=np.random.normal(size=S)
        policies_random=create_random_policies(S)
        init_random=(V_random_,policies_random,S)   
        return(init_random)
    else:
        V_random_=np.ones(shape=S)
        same_policy_all_states=np.zeros(shape=(S,3));same_policy_all_states[:,1]=1
        init_not_random=(V_random_,same_policy_all_states,S)      
        return(init_not_random)


init_i=random_init_q(False)                       

S=100
V_1,pi_1,_=policy_evaluation(P,R,init_i);init_i=random_init_q(False)        
V_2,pi_2=policy_iteration(P,R,init_i,theta=0.0001);init_i=random_init_q(False)        
V_3,pi_3=policy_interation2(P,R,init_i,theta=0.0001);init_i=random_init_q(False)        
V_4,pi_4=policy_iteration3(P,R,init_i);init_i=random_init_q(False)        
V_5,pi_5=policy_iteration4(P,R,init_i,theta=0.0001);init_i=random_init_q(False)        
V_6,pi_6=value_iteration(P,R,S,init_i);init_i=random_init_q(False)        
init_2=(V_6,pi_6,S)
V_7,pi_7,_=policy_evaluation(P,R,init_2);init_i=random_init_q(False)     
V_8,pi_8,_=iterative_policy_evaluation(P,R,init_i)

###  PREGUNTA ????????? ##############

plt.close('all')
#plt.plot(range(S),V_1,'r',)  # Qa2
#q_a_2=inv((np.eye(N)-P2))@R2
#plt.plot(range(S),q_a_2,'c')
plt.plot(range(S),V_2,'k--')
plt.plot(range(S),V_3,'y')
#plt.plot(range(S),V_4,'--g')
#plt.plot(range(S),V_5,'--b')
#plt.plot(range(S),V_6,'--m')
#plt.plot(range(S),init_i[0],'c')
#plt.plot(range(S),V_7,'g')
#plt.plot(range(S),V_8,'k')
plt.xlabel('states')
plt.ylabel('Value function')

###### pregunta auotpretungia######### deberían ser iguales ?
#init=random_init_q(random=True)
#init1=init
#init2=init
#V1,pi1,S=iterative_policy_evaluation(P,R,init1,theta=0.0001)
#V2,pi2,S=policy_evaluation(P,R,init2)
#plt.close('all')
#plt.plot(range(S),V1,'b')
#plt.plot(range(S),V2,'--r')
#plt.plot(range(S),init[0],'k')

