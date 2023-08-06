import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# log-likelihood function of Bernoulli
def log_like(x):
    p=1/np.mean(x)
    if min(p,1-p)<0.0005:
        return 0, p
    return sum([(xi-1)*np.log(1-p)+np.log(p) for xi in x]), p


# Dynamic Programming: multiple change-point detection for a Binary sequence 
def DP_segmentation(bin_process, thresh):
    event_indx=np.where(bin_process==1)[0]
    bin_process1=np.concatenate([[0],event_indx+1,[len(bin_process)+1]]) # time point from 1 to N
    R=np.diff(bin_process1)  # waiting time starts from 1
    R_prime=np.concatenate([[1000],R,[1000]])

    kick_list=np.argsort(R_prime)[:(-2)] # argsort: from smallest to largest

    set1=list(range(len(R_prime)))
    set2=[]
    cp_list=[] #record change points
    for m in kick_list:
        if m not in set1:
            continue

        m_indx=set1.index(m)
        left=set1[m_indx-1]
        right=set1[m_indx+1]

        if m in set2:
            m_indx2=set2.index(m)
            if (left in set2) and (right in set2):
                if m_indx2%2==0:
                    set2.remove(m)
                    set2.remove(left)
                    set1.remove(left) # set1 remove as well
                else:
                    set2.remove(m)
                    set2.remove(right)
                    set1.remove(right) # set1 remove as well
            elif left in set2:
                set2[m_indx2]=right

            elif right in set2:
                set2[m_indx2]=left

            else:
                print('error')
        else:
            delta=right-left
            if delta>thresh:  # threshold
                if (left in set2) and (right in set2):
                    set2.remove(left)
                    set2.remove(right)
                    set1.remove(left)  # set1 remove as well
                    set1.remove(right) # set1 remove as well

                elif left in set2:
                    set2[set2.index(left)]=right
                    set1.remove(left)

                elif right in set2:
                    set2[set2.index(right)]=left
                    set1.remove(right)

                else:
                    set2.append(left)
                    set2.append(right)
                    set2.sort()
        set1.remove(m)

        if (len(set2)>0) and (set2 not in cp_list):
            cp_list.append(tuple(set2))

    neg_loglike={} # record -2*logLikelihood
    change_point={}# record change points in original scale (event_indx)
    p_estimation={}# record estimated parameter in geometric
    
    for cp in cp_list:
        cp0=[0]+list(cp)+[len(R)]
        cp1=np.unique([len(R) if x>=len(event_indx) else x for x in cp0]) ## correct the length
        
        # calculate logLikelihood & Number of segments
        logL=0
        Num=0
        p_hat_rec=[]
        for i in range(0,len(cp1)-1):
            log_likelihood, p_hat = log_like(R[cp1[i]:cp1[i+1]])
            logL+=log_likelihood
            Num+=1
            p_hat_rec.append(p_hat) ###
            
       # update results
        if Num in neg_loglike.keys():
            if -2*logL < neg_loglike[Num]:
                neg_loglike[Num] = -2*logL
                change_point[Num]= [event_indx[xi] for xi in cp1[1:(-1)]]
                p_estimation[Num]= p_hat_rec ###
        else:
            neg_loglike[Num] = -2*logL
            change_point[Num]= [event_indx[xi] for xi in cp1[1:(-1)]]
            p_estimation[Num]= p_hat_rec ###
            
    return neg_loglike, change_point, p_estimation


# Model selection: segment the dimension space via K-means
# Search for change points for each Binary sequence, respectively
# Then aggregate the voting results with weights.
def Change_Point_Detection(data, K, nn_rate, crit='AIC'):
    km=KMeans(n_clusters=K, n_init=50).fit(data)
    # for each center, find its k neareast neighbors (param: nn)
    nn=int(data.shape[0]*nn_rate)
    neigh = NearestNeighbors(n_neighbors=nn).fit(data)
    neigh_set=neigh.kneighbors(km.cluster_centers_, return_distance=False)
    
    permission_matrix=np.zeros((data.shape[0],K))
    score=[]
    for v in range(K):

        bin_process=np.zeros(data.shape[0])
        bin_process[neigh_set[v]]=1
        
        IC_init=10000
        change_point_init=-1
        p_estimation_init=-1

        for th in np.arange(2,int(data.shape[0]*nn_rate)): # running over the tuning parameter theta
            neg_loglike, change_point, p_estimation = DP_segmentation(bin_process, thresh=th)
            
            # record the best candidate with best Information Criteria (IC)
            for k in neg_loglike.keys():
                if crit=='AIC':
                    IC = neg_loglike[k] + 2*( 2*len(change_point[k])+1 )
                elif crit=='BIC':
                    IC = neg_loglike[k] + np.log(len(data))*( 2*len(change_point[k])+1 )
                else: # Hannan-Quinn
                    IC = neg_loglike[k] + 2*np.log(np.log(len(data)))*( 2*len(change_point[k])+1 )
                
                if IC < IC_init: 
                    IC_init=IC
                    change_point_init=change_point[k]
                    p_estimation_init=p_estimation[k]

        if change_point_init==-1:
            score.append(10000)
            continue
        score.append(IC_init)

        L=[0]+change_point_init+[data.shape[0]]
        L_d=np.diff(L)
        p_vector=np.concatenate([np.repeat(p_estimation_init[x],L_d[x]) for x in range(len(L_d))])
        permission_matrix[:,v]=p_vector

    # simple feature importance weighting 
    score = np.array(score)
    weight = 1-(score-min(score))/(max(score)-min(score))
    weight/=sum(weight)

    d0={}
    for j in range(permission_matrix.shape[1]):
        rec=[]
        position=np.sort(neigh_set[j])
        cp=np.where(np.diff(permission_matrix[:,j])!=0)[0]+1
        for cps in cp:
            i=np.where(position==cps)[0]
            if i==0:
                rec+=list(np.arange(position[i],position[i+1]))
            elif i==(len(position)-1):
                rec+=list(np.arange(position[i-1],position[i]))
            else:   
                rec+=list(np.arange(position[i-1],position[i+1]))
        rec=np.unique(rec)
        for c in rec:
            d0[c]=d0.get(c,0)+weight[j] # weighted change-points voting

    return d0
