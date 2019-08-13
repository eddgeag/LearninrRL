#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:55:12 2019

@author: edmond
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import isclose
import gym 
import time as time
from collections import defaultdict
import collections
import sys
from lib import plotting
from lib.envs.gridworld import GridworldEnv
import pprint
from tqdm import tqdm
import json

''' enviraoment golf, deterministic and probabilistib'''

def play_golf(policy_state,s_g): 
    

    pi,state,S=policy_state
    action=np.argmax(pi)
    
#    action,state,S=policy_state
    

    
    a1,a2,a3=(0,1,2)
    l1,l2,l3=(2,6,np.round(S/2)-2)
    r1,r2,r3,r_g=(-1,-4,-S/4,0)
    #### first action and P#####
    if action == a1:
        embrace=2
        prob=1/2
        long=2

    ### second action and P#####
    if action ==a2:
        embrace=4
        prob=1/4
        long=6

    
    ##### third action and P#####
    if action == a3:
        embrace=5
        prob=1/5
        long=np.round(S/2)-2
       
    inicio=state+long
    inicio,fin=int((inicio-np.round(embrace/2))%S),int((inicio+np.round(embrace/2))%S)   
#    print(inicio,fin,state,'hhere2')
    aux=np.asarray(range(S))
    if state == s_g:
#            print('aqui3')
            inicio=inicio-1
            
    if inicio <  s_g and fin > inicio:
#        print('aqui4')
#        print(aux[inicio:fin])
        next_states=aux[inicio:fin]
        len_aux=len(next_states)
        idx=np.random.randint(0,len_aux)
        next_state=next_states[idx]

            
    elif inicio >= s_g and fin > inicio:
#        print('aqui6')

        inicio=inicio
        next_states=aux[inicio:fin]
#        print(next_states)
        len_aux=len(next_states)
        idx=np.random.randint(0,len_aux)
        next_state=next_states[idx]

            

    elif inicio >= s_g and fin < inicio:
#        print('aqui8')


        if len(aux[inicio:])< embrace:
#            print('aqui9')

#                print('aqui',inicio)

            fin=embrace-len(aux[inicio:])
            if  fin < embrace:
#                 print('aqui10')

                 next_states_1=aux[inicio:]
                 next_states_2=aux[:fin]
                 next_states=np.r_[next_states_1,next_states_2]
                 len_aux=len(next_states)
                 idx=np.random.randint(0,len_aux)
                 next_state=next_states[idx]
        elif len(aux[inicio:])== embrace:
#                print('aqui11')

                next_states=aux[inicio:]
                len_aux=len(next_states)
                idx=np.random.randint(0,len_aux)
                next_state=next_states[idx]
        if inicio >= S:
#                    print('aqui11')

                next_states=aux[:fin]
                    
                len_aux=len(next_states)
                idx=np.random.randint(0,len_aux)
                next_state=next_states[idx]
    elif inicio > fin:
        if len(aux[inicio:])< embrace:
#            print('aqui9')

#                print('aqui',inicio)

            fin=embrace-len(aux[inicio:])
            if  fin < embrace:
#                 print('aqui10')

                 next_states_1=aux[inicio:]
                 next_states_2=aux[:fin]
                 next_states=np.r_[next_states_1,next_states_2]
                 len_aux=len(next_states)
                 idx=np.random.randint(0,len_aux)
                 next_state=next_states[idx]
        elif len(aux[inicio:])== embrace:
#                print('aqui11')

                next_states=aux[inicio:]
                len_aux=len(next_states)
                idx=np.random.randint(0,len_aux)
                next_state=next_states[idx]
    else:
        
        raise ValueError(inicio,fin,S,'csa')
    
    
    reward=RewardFunction(action,state,S,s_g)
    if reward==0:
        done=True
    else:
        done=False
#        next_state,reward,action,state, done
#    print(next_state,reward,action,state, done)
    return(next_state,reward,action,state, done)

def env_golf(policy_state,play=True):
    pi,state,S=policy_state
    action=np.argmax(pi)
    a1,a2,a3=(0,1,2)
    #### first action and P#####
    if action == a1:
        embrace=2
        prob=1/2
        long=2

    ### second action and P#####
    if action ==a2:
        embrace=4
        prob=1/4
        long=6

    
    ##### third action and P#####
    if action == a3:
        embrace=5
        prob=1/5
        long=np.round(S/2)-2
       
#    print(action)
    inicio=state+long
    inicio,fin=int((inicio-np.round(embrace/2))%S),int((inicio+np.round(embrace/2))%S)   
#    print(inicio,fin,state,'hhere2')
    aux=np.asarray(range(S))
    if state == s_g:
#            print('aqui3')
            inicio=inicio-1
    if inicio <  s_g and fin > inicio:
#        print('aqui4')
#        print(aux[inicio:fin])
        next_states=aux[inicio:fin]
#        print(len(aux[inicio:fin+1])<embrace,len(aux[inicio:fin+1]),embrace)
        if len(aux[inicio:fin])<embrace:    
                    next_states=aux[inicio:fin+1]
#                    print('aqui5')

            
    elif inicio >= s_g and fin > inicio:
#        print('aqui6')

        inicio=inicio
        next_states=aux[inicio:fin]
#        print(next_states)
        if len(aux[inicio:fin])<embrace:
                     next_states=aux[inicio:fin+1]
#                     print('aqui7')

            

    elif inicio >= s_g and fin < inicio:
#        print('aqui8')


        if len(aux[inicio:])< embrace:
#            print('aqui9')

#                print('aqui',inicio)

            fin=embrace-len(aux[inicio:])
            if  fin < embrace:
#                 print('aqui10')

                 next_states_1=aux[inicio:]
                 next_states_2=aux[:fin]
                 next_states=np.r_[next_states_1,next_states_2]
        elif len(aux[inicio:])== embrace:
#                print('aqui11')

                next_states=aux[inicio:]
        if inicio >= S:
#                    print('aqui11')

                    next_states=aux[:fin]
    elif inicio==s_g:
#        print('aqui11')

        pass
    else:
        raise ValueError('cosa')

    next_states_=next_states
    if inicio == s_g:
#        print('cosasss')
        probabiliy=[0]
        rewards=[0]
        next_states=[0]
    else:
#        print(action,state,S,RewardFunction2(action,state,S),type(np.ones(shape=long)))
#        print(next_states_,state)
        rewards=RewardFunction(action,state,S)*np.ones(shape=len(next_states))
        probabiliy=prob*np.ones(shape=len(next_states))
        pos=np.random.randint(0,len(rewards))
        
        if play==True:
            if next_states[pos] ==s_g:
                done=True
            else:
                done=False
            print(type(next_states[pos]),type(RewardFunction(action,state,S)),type(done))
            return(next_states[pos],RewardFunction(action,state,S),done)
        else:
            return([probabiliy,next_states_,rewards])

''' common  functions'''
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

def random_init_v(random,S=100):
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
def random_init_q(random,S=100,n_acions=3):
    if random==True:
        Q_random_=np.random.normal(size=(S,n_acions))
        policies_random=create_random_policies(S)
        init_random=(Q_random_,policies_random,S)   
        return(init_random)
    else:
        Q_random_=np.ones(shape=(S,n_acions))-500
        same_policy_all_states=np.zeros(shape=(S,3));same_policy_all_states[:,1]=1
        init_not_random=(Q_random_,same_policy_all_states,S)      
        return(init_not_random)

def RewardFunction(a,s,S,s_g=50):
    if s == s_g:
        return(0)
    elif s!=s_g and a == 0:
        return(-1)
    elif s!=s_g and a == 1:       
        return(-4)
    elif s!=s_g and a ==2:
        return(-S/4)

        
def reward_vector(S,a,s_g):
    r=[]
    for si in range(S):
        value=RewardFunction(a,si,S)
        r.append(value)
    r=np.asarray(r).reshape(S,1)
    return(r)
def create_greedy_policy(state,S,nA,Q,epsilon):
    p=epsilon/nA
    cp=1-epsilon+p
    probs=np.ones(shape=nA)*p
    best_action=np.argmax(Q[state])
    probs[best_action]=cp
    ''' output las probabilidades para cada aaccion'''
    return(probs)

def create_random_uniform_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

######################## DP methods###########################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
    
def iterative_policy_evaluation(P,R,init,theta=0.0001):
    ''' es un policy iteration a traves de V'''
   
    Number_of_possible_action=3

    V_,pi_,S=init
    V=V_.copy()
    pi=pi_.copy()
    i=0
    policy_stable=False
    while policy_stable==False:
        delta=0
        for s,policy_ in enumerate(pi):
            v_=V[s].copy()
            
            aux_array=[]
            for a in range(Number_of_possible_action):
                aux=float(R[a,s,:]+P[a,s,:]@V)
                aux_array.append(aux)
            aux_array=np.asarray(aux_array)
            V[s]=np.sum(aux_array)
            diff=np.abs(v_-V[s])
            pi[s]=np.argmax(aux_array) # porque es determinista lo puedo hacer
#            print(diff,diff==0)

            delta=max([delta,diff])
            if  diff==0:
                policy_stable=True
        i=i+1
#        print('V',np.sum(V),'\ndiff',diff)
    V=np.asarray(V)
    return(V,pi,S)
        



    
def policy_iteration(P,R,init,theta=0.0001):
    
    
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
#                        print(R[a,s,:],P[a,s,:])
                        aux=R[a,s,:]+P[a,s,:]@V
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



def value_iteration(P,R,init,s_g=50,theta=0.0001):
    V_,pi_,S=init
    V=V_.copy()

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
#                print(V.shape)
                aux=float(R[a,s,:]+P[a,s,:]@V)
                aux_array.append(aux)
            V[s]=np.max(aux_array)
            diff=np.abs(v-V[s])
            delta=np.max([delta,diff])
            pi_s=np.zeros(shape=3)
            pi_s[np.argmax(aux_array)]=1
            pi[s,:]=pi_s
            if delta <= theta:

                print(i,'popo')
                return(V,pi)
                break
            i=i+1

def policy_evaluation(P,R,init,s_g,theta):
   

    V_,pi_,S=init
    V=V_.copy()
    V[s_g]=0
    pi=pi_.copy()
    i=0
    policy_stable=False
    iteracion=[]
    while policy_stable==False:
        delta=0
        for s,policy_ in enumerate(pi):
            v_=V[s].copy()
#            print('cosa',np.where(policy_ !=0)[0])
            a=int(np.where(policy_ !=0)[0])
            V[s]=float(R[a,s,:]+P[a,s,:]@V)
            diff=np.abs(v_-V[s])
            delta=max([delta,diff])
            if delta <= theta :
                policy_stable=True
                iteracion.append([i,delta,diff])
                
        i=i+1
#        print('V',np.sum(V),'\ndiff',diff)
    V=np.asarray(V)
    return(V,pi,S,iteracion)


def policy_eval_version_internet(init, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        es el "ambiente" de golf, donde se saben todas las probabilidades"
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V_,policy,S=init
    V=V_.copy()
    itera=0
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(S):
            v = V[s].copy()
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                prob,next_state,reward=env(a,s,S)
                for   i in range(len(prob)):
#                    print(i,'itera',len(prob) ,len(reward),len(next_state),a,s,S)
                    # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    v += action_prob * prob[i] * (reward[i] + discount_factor * V[next_state[i]])
                    
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
        itera=itera+1
#        print(itera,'lolo')
    return np.array(V)      

def policy_evaluation_Q(init,P,R,theta):
    Q_,pi_,S=init
    Q=Q_.copy()
    Q[s_g]=0
    pi=pi_.copy()
    i=0
    policy_stable=False
    iteracion=[]
    while policy_stable==False:
        delta=0
        for s,policy_ in enumerate(pi):
            for a in range(Q.shape[1]):
                q_=Q[s,s].copy()
                Q[s]=float(R[a,s,:]+P[a,s,:]@Q)
                diff=np.abs(q_-Q[s,a])
                delta=max([delta,diff])
                if delta <= theta :
                    policy_stable=True
                    iteracion.append([i,delta,diff])
                    
        i=i+1
#        print('V',np.sum(V),'\ndiff',diff)
    Q=np.asarray(Q)
    return(Q,pi,S,iteracion)
##############################################################################
##############################################################################
##############################################################################
##############################################################################

''' MC methdos'''
    
def monte_carlo_es(episodes,init,s_g): ### policy improvement throtuh Q
    n_actions=3
    init=random_init_q(False)
    Q_,pi_,S=init
    Q,pi=Q_.copy(),pi_.copy() # qnot related v and pi
#    print(Q.shape)
    rewards=[]
    for s in range(S):
        rewards.append(([],[],[]))
        
    for e in range(episodes):
#        print(i_episode)
    # Print out which episode we're on, useful for debugging.
        if e % 1000 == 0:
            print("\rEpisode {}/{}.".format(e, episodes), end="")
#        print(e)
        initial_state=np.random.randint(S)
        initial_action=np.random.randint(n_actions)
        pi[initial_state]=0
        pi[initial_state][initial_action]=1
        policy_state_action=[pi[initial_state],initial_state,S]
        episode=[]
        done=False
        T=100
        #generate episode
        for t in range(T):
            
            next_state,reward,action,state, done= play_golf(policy_state_action,s_g=50)
            episode.append([next_state,reward,(action,state), done])
            if done == True:

                break
            state=next_state
            policy_state_action=(pi[state],state,S)
                
        #### end generate episode
        T=len(episode)
        state_action_in_episode=[]
        for t in range(T):
            _,_,action_state,_=episode[t]
            state_action_in_episode.append(action_state)
        state_action_in_episode=set(state_action_in_episode)
        for (action_i,state_i) in state_action_in_episode:
#            print((action_i,action_i))
            # Find the first occurance of the state in the episode
            for i,x in enumerate(episode):
#                print(x[2])
                if x[2]==(action_i,state_i):
                    first_occurence_idx=i
                    break
            # Sum up all rewards since the first occurance
            G=0
            for i,x in enumerate(episode[first_occurence_idx:]):
                r=x[1]
                G=G+r
                rewards[state_i][action_i].append(G)

            # Calculate average return for this state over all sampled episodes
            
#            print(rewards[state],action_i)
            
#            rewards[state_i][action_i].append(G)
            Q[state_i][action_i] = np.mean(rewards[state_i][action_i])
#        print(Q)
        for s in range(S):
            pi[s]=0
            argmax=np.argmax(Q[s])
            pi[s][argmax]=1
            
    return(Q,pi)




def First_visit_MC_prediction(episodes,init,s_g): ## policy evaluacion through V
    V_,pi_,S=init
    V,pi=V_.copy(),pi_.copy()
    Reward=[]
    for s in range(S):
        Reward.append([])
#    s_i=np.random.randint(S)
    for  e in range(episodes):
        if e % 100 == 0:
            print("\rEpisode {}/{}.".format(e, episodes), end="")
            sys.stdout.flush()

#        print(e)
        done=False

        episode=[]
        state = np.random.randint(S)
        done=False
        policy_state=pi[state],state,S
        T=100
        for t in range(T):
            policy_state=pi[state],state,S

            next_state,reward,action,state, done=  play_golf(policy_state,s_g)#input_policy,state
            action=np.argmax(pi[state])
            episode.append((next_state,reward,action,state, done))
            if done==True:
                state = next_state
                T=t
                break    
            state = next_state    
            
            
#       OJO LEER BIEN ESTE PEDAZO DE COGIDO EN EL SUTTON EN EL PARRAFO ES TIRST VISIT   
        # Find all states the we've visited in this episode
        states_in_episode=[]
#        print(T,len(episode))
        for t in range(T):
            next_state,reward,action,state, done=episode[t]
            states_in_episode.append(state)
        
        states_in_episode = set(states_in_episode)
#        print(len(states_in_episode))
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            for i,x in enumerate(episode):
#                print(x)
                if x[3]==state:
                    first_occurence_idx=i
                    break
            # Sum up all rewards since the first occurance
            G=0
            for i,x in enumerate(episode[first_occurence_idx:]):
                r=x[1]
                G=G+r
                Reward[state].append(G)
            # Calculate average return for this state over all sampled episodes
#            Reward[state].append(G)
            V[state] = np.mean(Reward[state])

      

    return(V)  



    
def MC_FV_on_policy_control(episodes,init,epsilon,s_g): # #control FV improve policy and Q value
    Q_,pi_,S=init
    Q,pi=Q_.copy(),pi_.copy() # qnot related v and pi
#    print(Q.shape)
    rewards=[]
    for s in range(S):
        rewards.append(([],[],[]))    
    actions=[0,1,2]
    nA=Q.shape[1]
    T=100
    p=epsilon/nA
    pc=1-epsilon+p
    for i_episode in range(episodes):
#        print(i_episode)
    # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, episodes), end="")
            sys.stdout.flush()
        state=np.random.randint(S)

        # Generate an episode.
        episode=[]
        for t in range(T):
            probs=greedy_policy(state,S,nA,Q,epsilon) #generate arbitrary soft policy
            action=np.random.choice(actions, p=probs)
            pi_det=np.zeros(shape=nA)
            pi_det[action]=1
            policy_state_action=(pi_det,state,S)
            next_state,reward,action,state, done= play_golf(policy_state_action,s_g=50)
            episode.append([next_state,reward,(action,state), done])
            if done == True:
                break
            state=next_state
        ###look into all time steps of the episode
        state_action_in_episode=[]
        T=len(episode)
        for t in range(T):
            _,_,action_state,_=episode[t]
            state_action_in_episode.append(action_state)
        state_action_in_episode=set(state_action_in_episode)
        for (action_i,state_i) in state_action_in_episode:
#            print((action_i,action_i))
            # Find the first occurance of the state in the episode
            for i,x in enumerate(episode):
#                print(x[2])
                if x[2]==(action_i,state_i):
                    first_occurence_idx=i
                    break
            # Sum up all rewards since the first occurance
            G=0
            for i,x in enumerate(episode[first_occurence_idx:]):
                r=x[1]
                G=G+r
                rewards[state_i][action_i].append(G)
                
            # Calculate average return for this state over all sampled episodes
            #average

            
#            print(rewards[state_i][action_i])
            Q[state_i][action_i] = np.mean(np.asarray(rewards[state_i][action_i]))
            A=np.argmax(Q[state_i])
            for a in actions:
                if a == A:
                    pi[state_i][action_i]=pc
                else:
                    pi[state_i][action_i]=p
    return(Q,pi)

def mc_control_off_policy_w_importance_sampling(init,num_episodes, behavior_policy,s_g ,discount_factor=1.0):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Gamma discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
        
        
    ATENCION: Si se inserta un array de policies greegy cada una, hará un improvement
    si se inserta una "fija" tedríá que concluir con resultados de Q
    MODIFICAR PARA TENER ESTAS DOS OPCIONES !!! Y LA TERMINACION, NO PONER 1 SINO
    LA PONER LA POLICY(A|S) !!! 
    TAMBIEN IMPLEMENTAR ORDINARY SAMPLING !
    """
    Q,pi,S=init
    nA=pi.shape[1]
    # The final action-value function.
    # A dictionary that maps state -> action values

    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C=np.zeros(shape=(S,nA))
    
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)
        
    for i_episode in range(0, num_episodes):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state=np.random.randint(S)
        for t in range(100):
            # Sample an action from our policy
            
            pi_random=np.zeros(nA)
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            pi_random[action]=1
            policy_state_action=pi_random,state,S
            next_state,reward,action,state, done= play_golf(policy_state_action,s_g)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            # Update the total reward since step t
            G = discount_factor * G + reward
            # Update weighted importance sampling formula denominator
            C[state][action] += W
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            # If the action taken by the behavior policy is not the action 
            # taken by the target policy the probability will be 0 and we can break
#            print(target_policy(state))
            if action !=  np.argmax(target_policy(state)):
                break
            W = W * 1./behavior_policy(state)[action]
        
        
    return Q, target_policy
def policy_ealuation_MC_off_policy(episodes,init,s_g=50):
    ''' TIENE QUE DAR EL MISMO RESULTADO QUE LA ANTERIOR, PONIENDOLE A LA 
    ANTERIOR UN ARRAY DE POLICIES FIJAS Y DETERMINISTICAS'''
    
    Q_,pi,S=init
    Q=Q_.copy()
    target_policy=pi.copy()
    nA=pi.shape[1]
    C=np.zeros(shape=(S,nA))
    actions=[0,1,2]
    T=100
    for e in range(episodes):
#        print('episodeio',e)
    # Print out which episode we're on, useful for debugging.
#        print(e)
     if e % 1000 == 0:
        print("\rEpisode {}/{}.".format(e, episodes), end="")
        sys.stdout.flush()

        episode = []
        state=np.random.randint(S)
        for t in range(T):
            probs = policy_random(nA)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            pi_random=np.zeros(shape=nA)
            pi_random[action]=1
            policy_state_action=pi_random,state,S
            next_state,reward,action,state, done= play_golf(policy_state_action,s_g)
            episode.append((next_state,reward,action,state, done))
            if done:
                break
            state = next_state
        
        
        
        G=0
        W=1
        
        for t in range(len(episode))[::-1]:
#            print('iteration aqui wey',t)
#            print(len(episode),episode[t])
            next_state,reward,action,state, done=episode[t]
#            print(next_state,reward,action,state, done)
            G=G+reward
            C[state][action]=C[state][action]+W
            Q[state,action]=Q[state,action]+(W/C[state,action])*(G-Q[state,action])
            prob_pi=target_policy[state][action]
            probs = policy_random(nA)
            prob_b=probs[action]
            rho=prob_pi/prob_b
            W=W*rho
            if W==0 :
                break
        
    return(Q,target_policy)

def First_visit_MC_prediction_modified(episodes,init,theta,s_g=50):
    ''' Policy Evaluacion pero más rápida. COMPROBAR !!! ( Y ES UN EJERCICIO)'''
    
    V_,pi_,S=init
    V,pi=V_.copy(),pi_.copy()
    Reward=[]
    for s in range(S):
        Reward.append([])
#    s_i=np.random.randint(S)
    for  e in range(episodes):
#        print(e)
        if e % 100 == 0:
            print("\rEpisode {}/{}.".format(e, episodes), end="")
            sys.stdout.flush()

#        print(e)
        done=False

        episode=[]
        state = np.random.randint(S)
        done=False
        policy_state=pi[state],state,S
        T=100
        for t in range(T):
            policy_state=pi[state],state,S

            next_state,reward,action,state, done=  play_golf(policy_state,s_g)#input_policy,state
            action=np.argmax(pi[state])
            episode.append((next_state,reward,action,state, done))
            if done==True:
                state = next_state
                T=t
                break    
            state = next_state    
            
            
#       OJO LEER BIEN ESTE PEDAZO DE COGIDO EN EL SUTTON EN EL PARRAFO ES TIRST VISIT   
        # Find all states the we've visited in this episode
        states_in_episode=[]
#        print(T,len(episode))
        for t in range(T):
            next_state,reward,action,state, done=episode[t]
            states_in_episode.append(state)
        
        states_in_episode = set(states_in_episode)
#        print(len(states_in_episode))
        
        for state in states_in_episode:
            v=V[state]
            # Find the first occurance of the state in the episode
            for i,x in enumerate(episode):
#                print(x)
                if x[3]==state:
                    first_occurence_idx=i
                    break
            # Sum up all rewards since the first occurance
            G=0
            delta=0
            policy_stable=False
            while policy_stable==False:
                for i,x in enumerate(episode[first_occurence_idx:]):
                    r=x[1]
                    G=G+r
                    Reward[state].append(G)
                # Calculate average return for this state over all sampled episodes
    #            Reward[state].append(G)
                N=len(Reward[state])
                V[state]=np.min([v,V[state]])
                V[state] = V[state]+(1/N)*(G-V[state])
                v=V[state]
                diff=np.abs(v-V[state])
#                print(diff)
                delta=max([delta,diff])
                if delta <= theta :
                    policy_stable=True
#                    iteracion.append([i,delta,diff])


'''variables'''
Number_of_States=100
s_g=50

S=Number_of_States
#### first action and P#####
a1=1
embrace_1=2
prob_1=1/2
long_1=2
#long_1=0

### second action and P#####
a2=2
embrace_2=4
prob_2=1/4
long_2=6
#long_2=0

##### third action and P#####
a3=3
embrace_3=5
prob_3=1/5
long_3=np.round(S/2)-2
#long_3=0

######## Enviroment in we will play #########
P1=make_trans_prob(long_1,embrace_1,prob_1,S,s_g); P1[s_g,:]=0;# P1[s_g,s_g]=1
R1=reward_vector(S,0,s_g)


P2=make_trans_prob(long_2,embrace_2,prob_2,S,s_g); P2[s_g,:]=0; #P2[s_g,s_g]=1
R2=reward_vector(S,1,s_g)

R3=reward_vector(S,2,s_g)
P3=make_trans_prob(long_3,embrace_3,prob_3,S,s_g);P3[s_g,:]=0; #P3[s_g,s_g]=1


P=np.array([P1,P2,P3])
R=np.array([R1,R2,R3])


###############three diferents policies for all states #### NO ARBITRARY ######+

##### 3 policies: for action 1,2,3  for all states respectivmente ######


P1=make_trans_prob(long_1,embrace_1,prob_1,S,s_g); P1[s_g,:]=0;# P1[s_g,s_g]=1
R1=reward_vector(S,0,s_g)


P2=make_trans_prob(long_2,embrace_2,prob_2,S,s_g); P2[s_g,:]=0; #P2[s_g,s_g]=1
R2=reward_vector(S,1,s_g)

R3=reward_vector(S,2,s_g)
P3=make_trans_prob(long_3,embrace_3,prob_3,S,s_g);P3[s_g,:]=0; #P3[s_g,s_g]=1


P=np.array([P1,P2,P3])
R=np.array([R1,R2,R3])

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

##################  POR HACER SI TENGO TIEMPO en 7 dias  ########################################   
########## ejercicio ##############################################################
#def mc_control__off_policy_modified( init,num_episodes, behavior_policy, discount_factor):
#    """
#    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
#    Finds an optimal greedy policy.
#    
#    Args:
#        env: OpenAI gym environment.
#        num_episodes: Number of episodes to sample.
#        behavior_policy: The behavior to follow while generating episodes.
#            A function that given an observation returns a vector of probabilities for each action.
#        discount_factor: Gamma discount factor.
#    
#    Returns:
#        A tuple (Q, policy).
#        Q is a dictionary mapping state -> action values.
#        policy is a function that takes an observation as an argument and returns
#        action probabilities. This is the optimal greedy policy.
#    """
#    
#    # The final action-value function.
#    # A dictionary that maps state -> action values
#    Q_,pi_,S=init
#    Q,pi=Q_.copy(),pi_.copy()
#    nA=pi.shape[1]
#    target_policy = create_greedy_policy(Q)
#    rewards=[]
#    for s in range(S):
#        rewards.append(([],[],[]))
#    rhos=[]
#    for s in range(S):
#        rhos.append(([],[],[]))
#                
#    for i_episode in range(1, num_episodes + 1):
#        # Print out which episode we're on, useful for debugging.
##        if i_episode % 100 == 0:
##            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
##            sys.stdout.flush()
#
#        # Generate an episode.
#        # An episode is an array of (state, action, reward) tuples
#        episode = []
#        state = np.random.randint(S)
#        for t in range(100):
#            # Sample an action from our policy
#            probs = behavior_policy(state)
#            action = np.random.choice(np.arange(len(probs)), p=probs)
#            pi_random=np.zeros(shape=nA)
#            pi_random[action]=1
#            policy_state_action=pi_random,state,S
#            next_state,reward,action,state, done= play_golf(policy_state_action,s_g)
#            episode.append((next_state,reward,action,state, done))
#            if done :
#                break
#            state = next_state
#        
#        
#        
#
#            if done:
#                break
#            state = next_state
#        
#        # Sum of discounted returns
#        # The importance sampling ratio (the weights of the returns)
#        G=0
#        # For each step in the episode, backwards
#        for t in range(len(episode)):
#            
#            next_state,reward,action,state, done=episode[t]
#            G = discount_factor* G + reward
##            print(discount_factor,((discount_factor)**(t)) )
#            prob_pi_v=target_policy(state)
#            prob_pi=prob_pi_v[action]
#            probs = policy_random(nA)
#            prob_b=probs[action]
##            print(prob_pi_v[action],prob_pi_v)
#            rho=prob_pi/prob_b
#            rewards[state][action].append(G)
#            rhos[state][action].append(rho)
#            if G==0 and rho ==0:
##                print(G,rho)
#                break
#            else:
#
#                vector_rewards=np.asarray(rewards[state][action])
#                vector_rhows=np.asarray(rhos[state][action])
#                product=vector_rewards * vector_rhows
#                N=np.mean(product)
#                D=np.mean(vector_rhows)
#                Q[state][action]=N/D
##                print('holi')
##            print(N/D)1
#                
#
#    return Q, target_policy
#
#'''NOTA #17 cifras decimales  para igualar a un numero en '''     ###
#''' ejercicio sin poder replicar resultados del sutton '''
#def miniMDP(state,policy):
#    
#    Terminal=True
#    if state != Terminal:
#        action=np.argmax(policy)
#        if action ==0: #LEFT
#            probs=[0.9,0.1]
#            action2=np.random.choice([0,1],p=probs)
#            if action2 == 0:
#                reward=0
#                next_state=state
#                step=[next_state,state,action,reward]
#            elif action2 ==1:  #RIGHT
#                reward=1
#                next_state=Terminal
#                step=[next_state,state,action,reward]
#                
#        elif action ==1: #RIGHT
#    
#            reward=0
#            next_state=Terminal
#            step=[next_state,state,action,reward]
#        else:
#            raise ValueError('caca')
#    
#        
#    else:
#        step=[True,True,action,reward]
#        
#    return(step)
#
#
#def policy_ealuation_MC_off_policy_modified(episodes,init):
#    V_,pi,S=init
#    V=V_.copy()
#    target_policy=pi.copy()
#
#    nA=target_policy.shape[0]
#    T=100
#    C=np.zeros(shape=(S))
#    rhos=[]
#    Reward=[]
#    for s in range(S):
#        Reward.append([])    
#    for s in range(S):
#        rhos.append([])
#    for e in range(episodes):
##        print('episodeio',e)
##     Print out which episode we're on, useful for debugging.
##        print('episodeio',e)
##     if e % 10000 == 0:
##        print("\rEpisode {}/{}.".format(e, episodes), end="")
##        sys.stdout.flush()
#
#        episode = []
#        state=0
#        for t in range(T):
#            probs = policy_random(nA)
#            action = np.random.choice(np.arange(len(probs)), p=probs)
##            print('action',action)
#            pi_random=np.zeros(shape=nA)
#            pi_random[action]=1
#            ### make a step
#            next_state,state,action,reward=miniMDP(state,pi_random)
#            episode.append((next_state,state,action,reward))
#            state=next_state
#            if next_state==True:
#                break
#
#
#        
#        
#        
##        print(episode)
#        states_in_episode=[]
##        print(T,len(episode))
#        for t in range(len(episode)):
#            next_state,state,action,reward=episode[t]
#            states_in_episode.append(state)
#        
#        states_in_episode = set(states_in_episode)
##        print(len(states_in_episode))
#        for state in states_in_episode:
#            # Find the first occurance of the state in the episode
#            for i,x in enumerate(episode):
##                print(x)
#                if x[1]==state:
#                    first_occurence_idx=i
#                    break
#            # Sum up all rewards since the first occurance
#            G=0
#            W=1
#
#            
#            for i,x in enumerate(episode[first_occurence_idx:]):
##                print(i,x,'acaaa')
#                r=x[3]
#                G=G+r
##                print(state,G)
#                Reward[state].append(G)
#                prob_pi=target_policy[action]
#                probs = policy_random(nA)
#                prob_b=probs[action]
#                rho=prob_pi/prob_b
#                W=W*rho
#                rhos[state].append(W)
#                product=np.asarray(Reward[state])*np.asarray(rhos[state])
##                print(np.mean(product))
#                V[state] = np.mean(product)
#                if W==0:
#                    break
#                
#
##                RES.append(G)
#                
#
#            # Calculate average return for this state over all sampled episodes
##            Reward[state].append(G)
#
##    print('end')
#    return(V,target_policy)