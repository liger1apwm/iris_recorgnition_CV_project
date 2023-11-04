
import numpy as np

def CRR(true, pred): 
    correct = 0
    n = len(true)
    
    for i,j in zip(true, pred):
        if i == j :
            correct +=1
        else:
            correct +=0

    CRR = np.round(np.divide(correct, n)*100,2)

    return correct 
    


    







