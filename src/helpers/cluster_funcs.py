import numpy as np
import pickle as pkl
import time

from hkmeans import HKMeans

def run_multiple_report_best_new(ks, x, directory, max_counter =  25):
    
    best_wcss  = []
    best_mod = []
    for k in ks:
        print('k = ' + str(k))
        prev_best_wcss = np.inf
        counter = 1
        for i in range(400):
            print("i = {}, counter = {}".format(i, counter))
            mod =  HKMeans(n_clusters = k, max_iter = 10, tol = 0.01, n_init = 25, n_jobs  =  10)
            mod.fit(x)
            print("Found solution WCSS: {}".format(mod.inertia_))
            if mod.inertia_ < prev_best_wcss:
                tmp_mod  = mod
                prev_best_wcss = mod.inertia_
                counter = 1
            else:
                counter += 1
            if counter % max_counter == 0:
                break
        best_wcss.append(tmp_mod.inertia_)
        best_mod.append(tmp_mod)
        
        print('saving temporary results')
        st = time.time()
        output = {'best_mod': best_mod, 'best_wcss' : best_wcss}
    
        filename = directory + 'wcss_new2.pkl'
        file = open(filename, 'wb')
        pkl.dump(output, file)
        file.close() 
        end = time.time()
        elapsed = end-st
        print('saving done,  it  took  {} seconds'.format(elapsed))
        
           
    output = {'best_mod': best_mod, 'best_wcss' : best_wcss}
    
    filename = directory + 'wcss.pkl'
    file = open(filename, 'wb')
    pkl.dump(output, file)
    file.close()   
    
    return best_mod, best_wcss

