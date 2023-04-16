import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from utils.mvecdf import MvECDF
from utils.utils import *

class MFDistributionLearning:
    
    """
    This class implements multifidelity distribution learning methods including the AETC-d and cvMDL algorithms. For cvMDL, the current version can only handle QoI dimension up to 3d.

    """
    def __init__(self, weight=None):
        
        self.alpha = 4  # regularization parameter (large values means no regularization)
        self.q = 0.05 # quantile truncation parameter for estimation in 1d
        self.N = 1000 # 1d exact integral pamameter
        self.NN = 50 # multidimensional integral mesh parameter
        
        # weight function is used in index computation
        if weight == None:    
            def weight_function(x):
                return 1
            self._weight = weight_function
        else:
            self._weight = weight
     
    def index_cvMDL(self, df_high, df_cv, cost):  
        """
        compute the indices k1, k2 for cvMDL model selection

        parameters:
        ----------------------------------------------------
        df_high: high-fidelity data (2d-array)
        df_cv: control variates data constructed from low-fidelity models (2d-array)
        cost: cost vector corresponding to the model (list or 1d-array) 

        """
        # exploitation cost
        c_ept = sum(cost)
        
        # get min/max of empirical CDF of df_high in each coordinate
        l_min = df_high.min(axis=0)
        l_max = df_high.max(axis=0)
        
        # get quantile of df_cv if df_cv is 1d
        if df_cv.shape[1] == 1:
            l_tau_lower = np.quantile(df_cv, self.q)
            l_tau_upper = np.quantile(df_cv, 1-self.q)
        
        # get max of df_high, df_cv columnwise
        max_high_cv = np.zeros(df_high.shape)
        for i in range(df_high.shape[1]):
            max_high_cv[:,i] = np.maximum(df_high[:,i], df_cv[:,i])
        
        # compute empirical CDF for df_high, df_cv, and their max(df_high, df_cv)
        a_Y = MvECDF(df_high)
        a_Z = MvECDF(df_cv)
        a_max = MvECDF(max_high_cv)
        
        # integrand of model selection indices
        def k_1_2_integrand(z):
            if type(z) != list and type(z) != np.ndarray:
                z = np.array([z])
            Y = np.array([1]*df_high.shape[0]) 
            Z = np.array([1]*df_high.shape[0])
            for i in range(df_high.shape[1]):
                Y *= np.less_equal(df_high[:,i], z[i])*1
                Z *= np.less_equal(df_cv[:,i], z[i])*1
            A = add_intercept(Z)
            res = model_ls(A, Y)
            
            # coompute k1 and k2
            k1 = np.mean(res['sd']**2)*self._weight(z)
            k2 = self._weight(z)*np.std(Y)**2 - k1
            return np.array([k1, k2])
        
        # whole range 
        def model_coef(z):
            t1 = a_Y.ecdf_eval(z)
            t2 = a_Z.ecdf_eval(z)
            t3 = a_max.ecdf_eval(z)
            # tail continuity estimation
            if df_high.shape[1] == 1:
                if t2*(1-t2) != 0:
                    alpha_z = (t3-t1*t2)/(t2*(1-t2))
                elif t2 == 0:
                    t1_l = a_Y.ecdf_eval(l_tau_lower)
                    t2_l = a_Z.ecdf_eval(l_tau_lower)
                    t3_l = a_max.ecdf_eval(l_tau_lower)
                    alpha_z = (t3_l-t1_l*t2_l)/(self.q*(1-self.q))
                else:
                    t1_r = a_Y.ecdf_eval(l_tau_upper)
                    t2_r = a_Z.ecdf_eval(l_tau_upper)
                    t3_r = a_max.ecdf_eval(l_tau_upper)
                    alpha_z = (t3_r-t1_r*t2_r)/(self.q*(1-self.q))
            else:
                if t2*(1-t2) == 0:
                    alpha_z = 0
                else:
                    alpha_z = (t3-t1*t2)/(t2*(1-t2))
            return t1-alpha_z*t2, alpha_z
        
        def model_coef_no(z):
            t1 = a_Y.ecdf_eval(z)
            t2 = a_Z.ecdf_eval(z)
            t3 = a_max.ecdf_eval(z)
            # tail continuity estimation
            if t2*(1-t2) == 0:
                alpha_z = 0
            else:
                alpha_z = (t3-t1*t2)/(t2*(1-t2))
            return t1-alpha_z*t2, alpha_z
        
        # compute k1 and k2; only work for 1/2/3D
        if df_high.shape[1] == 1:
            if df_high.shape[0] <= self.N:
                k1, k2 = integration(df_high.flatten(), df_cv.flatten(), k_1_2_integrand)
            else:
                k1, k2 = np.mean(np.array([k_1_2_integrand(z)*(l_max[0]-l_min[0]) for z in np.linspace(l_min[0], l_max[0], 1000)]), axis = 0)
        elif df_high.shape[1] == 2:
            L1 = np.linspace(l_min[0], l_max[0], self.NN)
            L2 = np.linspace(l_min[1], l_max[1], self.NN)
            d1, d2 = (l_max - l_min)/self.NN
            k1 = 0
            k2 = 0
            for i in range(self.NN):
                for j in range(self.NN):
                    k1_add, k2_add = k_1_2_integrand([L1[i], L2[j]])
                    k1 += k1_add*d1*d2
                    k2 += k2_add*d1*d2
        elif df_high.shape[1] == 3:
            L1 = np.linspace(l_min[0], l_max[0], self.NN)
            L2 = np.linspace(l_min[1], l_max[1], self.NN)
            L3 = np.linspace(l_min[2], l_max[2], self.NN)
            d1, d2, d3 = (l_max - l_min)/self.NN
            k1 = 0
            k2 = 0
            for i in range(self.NN):
                for j in range(self.NN):
                    for l in range(self.NN):
                        k1_add, k2_add = k_1_2_integrand([L1[i], L2[j], L3[l]])
                        k1 += k1_add*d1*d2
                        k2 += k2_add*d1*d2
        else:
            raise TypeError("Data should be 1d, 2d, or 3d.") 
        res = {'k1': k1, 'k2': c_ept*k2, 'rho': model_coef, 'rho_no': model_coef_no}
        return res        
    
    def cvMDL_pars(self, df_low, df_high, cost, tol_size):
        """
        get oracle parameters/for simulation purpose only

        parameters:
        ----------------------------------------------------
        df_low: a dictionary
        df_high: high-fidelity data (2d-array)
        cost: cost vector corresponding to the model (list or 1d-array)
        total size: an integer 
        """
         # exploration cost
        c_epr = sum(cost)
        
        # get exploration rate m and # low-fidelity models
        n_low = df_high.shape[0], len(df_low)
        model_list = get_model_list(n_low, tol_size)
        n_model = len(model_list)
        
        # get least squares for each model
        model_res = []
        model_names = list(df_low.keys())
        for S in model_list:
            df_low_1 = df_low[model_names[S[0]]]
            if len(S)>1:
                for i in range(1,len(S)):
                    df_low_1 = np.concatenate((df_low_1, df_low[model_names[S[i]]]), axis=1)
            model_res.append(model_ls(add_intercept(df_low_1), df_high))
 
        # get df_cv for each model
        model_cv = [model_res[k]['fit'] for k in range(n_model)]
        
        # get k1/k2 for each model
        k_1_2 = [self.index_cvMDL(df_high = df_high, df_cv = model_cv[k], 
                cost = [cost[i] for i in model_list[k]]) for k in range(n_model)] 
        res = {' '.join([str(list(df_low.keys())[elem]) for elem in model]): (np.sqrt(c_epr*r['k1'])+np.sqrt(r['k2']))**2 \
               for model, r in zip(model_list, k_1_2)}
        return res
    
    def index_aetcd(self, sd, df_high, df_res, model, cost):
        """
        compute the indices k1, k2 for AETC-d model selection

        parameters:
        ----------------------------------------------------
        sd: standard deviation of noise (np.float)
        df_high: high-fidelity data (1d-array)
        df_res: residuals (1d-array)
        cost: cost vector corresponding to the model (list or 1d-array) 

        """
        s = len(model)
         
        # exploitation cost
        c_ept = sum(cost)
        
        # compute k1 and k2
        k1 = (2*np.sqrt(s+1)*sd+J1_eval(list(df_res)))**2
        k2 = c_ept*J1_eval(list(df_high))**2
        
        res = {}
        res['k1'] = k1
        res['k2'] = k2
        
        return res
 
    def model_select_cvMDL(self, df_low, df_high, cost, tol_size, budget):
        """
        select model S for cvMDL exploitation (cdf)

        parameters:
        ---------------------------------------------
        df_low: a dictionary
        df_high: a 2d-array with high-fidelity data
        cost: a list of cost parameters associated with the concatenated df_low and df_high 
        tol_size: maximum size of the low-fidelity models for linear regression
        budget: total budget for exploration and exploitation
        
        """
        
        # exploration cost
        c_epr = sum(cost)
        
        # get exploration rate m and # low-fidelity models
        m, n_low = df_high.shape[0], len(df_low)
        model_list = get_model_list(n_low, tol_size)
        n_model = len(model_list)
        
        # get least squares for each model
        model_res = []
        model_names = list(df_low.keys())
        for S in model_list:
            df_low_1 = df_low[model_names[S[0]]]
            if len(S)>1:
                for i in range(1,len(S)):
                    df_low_1 = np.concatenate((df_low_1, df_low[model_names[S[i]]]), axis=1)
            model_res.append(model_ls(add_intercept(df_low_1), df_high))
 
        # get df_cv for each model
        model_cv = [model_res[k]['fit'] for k in range(n_model)]
        
        # get k1/k2 for each model
        k_1_2 = [self.index_cvMDL(df_high = df_high, df_cv = model_cv[k], 
                cost = [cost[i] for i in model_list[k]]) for k in range(n_model)]
        
        # estimated optimal exploration rate
        model_opt = [budget/(c_epr + np.sqrt(c_epr*k['k2']/(k['k1'] + self.alpha**(-m)))) for k in k_1_2]
        # estimated optimal expploration rate at current time
        model_use = [max(m,s) for s in model_opt]
        # estimated optimal loss at current time
        model_risk = [k_1_2[s]['k2']/(budget-c_epr*model_use[s])+(k_1_2[s]['k1']+self.alpha**(-m))/model_use[s] for s in range(n_model)]
        # estimated optimal model
        n_opt = np.argmin(model_risk)

        # bisection exploration
        if model_opt[n_opt]>2*m:
            m_next = 2*m
        elif model_opt[n_opt]>m:
            m_next = int(np.ceil((m + model_opt[n_opt])/2))
        else:
            m_next = m
        
        # optimal exploitation cost
        c_opt = sum(cost[s] for s in model_list[n_opt])
        
        res = {'exp_rate': m_next, 'model': [model_names[s] for s in model_list[n_opt]],\
              'coef': model_res[n_opt]['Beta'], \
              'cost': c_opt, 'sample_rate': int(np.floor((budget-m_next*c_epr)/c_opt)),\
              'rho': k_1_2[n_opt]['rho'], \
              'rho_no': k_1_2[n_opt]['rho_no']}
        return res
        
    def model_select_aetcd(self, df_low, df_high, cost, tol_size, budget):
        """
        select model S for AETC-d exploitation (cdf)

        parameters:
        ----------------------------------------------------------
        df_low: a dictionary
        df_high: a 1d array with high-fidelity data
        cost: a list of cost parameters associated with the concatenated df_low and df_high 
        tol_size: maximum size of the low-fidelity models for linear regression
        budget: total budget for exploration and exploitation
        """
        
        # exploration cost
        c_epr = sum(cost)
        
        # get exploration rate m and # low-fidelity models
        m, n_low = df_high.shape[0], len(df_low)
        model_list = get_model_list(n_low, tol_size)
        n_model = len(model_list)
        
        # get least squares for each model
        model_res = []
        model_names = list(df_low.keys())
        for S in model_list:
            df_low_1 = df_low[model_names[S[0]]]
            if len(S)>1:
                for i in range(1,len(S)):
                    df_low_1 = np.concatenate((df_low_1, df_low[model_names[S[i]]]), axis=1)
            model_res.append(model_ls(add_intercept(df_low_1), df_high))
        
        # get k1/k2 for each model
        k_1_2 = [self.index_aetcd(sd = model_res[k]['sd']**2, df_high = df_high, df_res = model_res[k]['residuals'], 
                                  model = model_list[k], cost = [cost[i] for i in model_list[k]]) for k in range(n_model)]
        
        # estimated optimal exploration rate
        model_opt = [budget/(c_epr + (c_epr**2*k['k2']/(k['k1']+self.alpha**(-m)))**(1/3)) for k in k_1_2]
        # estimated optimal exploration rate at current time
        model_use = [max(m,s) for s in model_opt]
        # estimated optimal loss at current time
        model_risk = [(k_1_2[s]['k1']/model_use[s])**(1/2)+(k_1_2[s]['k2']/(budget-c_epr*model_use[s]))**(1/2) for s in range(n_model)]
        # estimated optimal model
        n_opt = np.argmin(model_risk)
        
        # bisection exploration
        if model_opt[n_opt]>2*m:
            m_next = 2*m
        elif model_opt[n_opt]>m:
            m_next = int(np.ceil((m + model_opt[n_opt])/2))
        else:
            m_next = m
        
        # optimal exploitation cost
        c_opt = sum([cost[s] for s in model_list[n_opt]])
        
        res = {'exp_rate': m_next, \
               'model': [model_names[s] for s in model_list[n_opt]], \
                'coeff': model_res[n_opt]['Beta'], 'noise_sd': model_res[n_opt]['sd'], \
               'sample_rate': int(np.floor((budget-m_next*c_epr)/c_opt)),\
                'residuals': model_res[n_opt]['residuals'],\
                 'cost': c_opt
              }
        return res
    
    def cvMDL(self, df_low, df_high, cost, tol_size, budget):
        """
        cvMDL

        parameters:
        -------------------------------------------------
        df_low: a dictionary
        df_high: a 2d-array with high-fidelity data
        cost: a list of cost parameters associated with the concatenated df_low and df_high 
        tol_size: maximum size of the low-fidelity models for linear regression
        budget: total budget for exploration and exploitation
        """
        c_epr = sum(cost)
        m = sum(x.shape[1] for x in list(df_low.values())) + 2
        low_keys = list(df_low.keys())
        if m>= budget/c_epr:
            return "The budget is too small."
        else:
            e = 1
            while e>0:
                df_low_temp = {key: df_low[key][0:m,:] for key in low_keys}
                res = self.model_select_cvMDL(df_low = df_low_temp, df_high = df_high[0:m,:], 
                                cost = cost, tol_size = tol_size, budget = budget)
                e = res['exp_rate'] - m
#                 print('Current exploration rate: {}'.format(m),
#                       'Current optimal model: {}'.format(res['model']), 
#                        'More exploration: {}'.format(e>0), sep = ", ")
                m = res['exp_rate']
                if m>df_high.shape[0]:
                    print('Get {} more exploration samples'.format(m-df_high.shape[0]))
                    break

            res['hf_data'] = df_high
            
            # get grid points where jumps occur
            model = res['model']
            df_low_opt = df_low[model[0]][:m,:]
            if len(model)>1:
                for i in range(1, len(model)):
                    df_low_opt = np.concatenate((df_low_opt, df_low[model[i]][:m,:]), axis=1)
            res['lf_data'] = add_intercept(df_low_opt)@res['coef']
            return res

    def aetcd(self, df_low, df_high, cost, tol_size, budget):
        """
        AETC-d

        parameters:
        -------------------------------------------------
        df_low: a dictionary
        df_high: a 2d-array with high-fidelity data
        cost: a list of cost parameters associated with the concatenated df_low and df_high 
        tol_size: maximum size of the low-fidelity models for linear regression
        budget: total budget for exploration and exploitation
        """
        c_epr = sum(cost)
        m = sum(x.shape[1] for x in list(df_low.values())) + 2
        low_keys = list(df_low.keys())
        if df_high.shape[1]>1:
            return "High-fidelity data needs to be scalar-valued."
        else:
            if m>= budget/c_epr:
                return "The budget is too small."
            else:
                e = 1
                while e>0:
                    df_low_temp = {key: df_low[key][0:m,:] for key in low_keys}
                    res = self.model_select_aetcd(df_low = df_low_temp, df_high = df_high[0:m,:], 
                                    cost = cost, tol_size = tol_size, budget = budget)
                    e = res['exp_rate'] - m
                    m = res['exp_rate']
                    if m>df_high.shape[0]:
                        print('Get {} more exploration samples'.format(m-df_high.shape[0]))
                        break
                return res

    def ecdf_cv(self, df_tt, res, tail=True):
        """
        cvMDL exploitation

        parameters:
        ----------------------------------------------------
        df_tt: a dictionary of exploitation data for estimation
        res: a dictionary given by the cvMDL output

        """
        model = res['model']
        if tail==True:
            rho = res['rho']
        else:
            rho = res['rho_no']
        n_ext = min(df_tt[model[0]].shape[0], res['sample_rate'])
        if n_ext < res['sample_rate']:
            print('Insufficient exploitation samples; {} more are needed'.format(res['sample_rate']-n_ext))
        df_ttt = df_tt[model[0]][0:n_ext,:]
        if len(model)>1:
            for i in range(1, len(model)):
                df_ttt = np.concatenate((df_ttt, df_tt[model[i]][0:n_ext,:]), axis=1)
        df = add_intercept(df_ttt)
        emulator = MvECDF(df@res['coef'])
        def cdf_cv(x):
            return max(min(emulator.ecdf_eval(x)*rho(x)[1] + rho(x)[0],1),0)
        return cdf_cv
     
    def ecdf_aetcd(self, df_tt, res, addnoise=True, inverse=False):
        """
        AETC-d exploitation

        parameters:
        ----------------------------------------------------
        df_tt: a dictionary of exploitation data for estimation
        res: a dictionary given by the AETC-d output

        """
        model = res['model']
        n_ext = min(df_tt[model[0]].shape[0], res['sample_rate'])
        if n_ext < res['sample_rate']:
            print('Insufficient exploitation samples; {} more are needed'.format(res['sample_rate']-n_ext))
        df_ttt = df_tt[model[0]][0:n_ext,:]
        if len(model)>1:
            for i in range(1, len(model)):
                df_ttt = np.concatenate((df_ttt, df_tt[model[i]][0:n_ext,:]), axis=1)
        df = add_intercept(df_ttt)
        emulator = (df@res['coeff']).flatten()
        if addnoise == False:
            cdf = ECDF(emulator)
        else:
            emulator += np.random.choice((res['residuals']).flatten().tolist(), size=n_ext, replace=True)
            cdf = ECDF(emulator)
        # get quantiles
        emulator.sort()
        x_min, x_max = emulator[0], emulator[-1]
        ecdf_emulator = np.linspace(1/len(emulator),1, len(emulator))
        cdf_inv = interp1d(ecdf_emulator, emulator)
        def quantile(x):
            if x<=1/len(emulator):
                return x_min
            elif x>=1:
                return x_max
            else:
                return cdf_inv(x)
        if inverse == False:
            return cdf
        else:
            return {'cdf': cdf, 'quantile': quantile}
    
    def ecdf_cv_1d(self, df_tt, res, inverse=False):
        """
        cvMDL exploitation with sorting

        parameters:
        ----------------------------------------------------
        df_tt: a dictionary of exploitation data for estimation
        res: a dictionary given by the cvMDL output

        """
        if res['hf_data'].shape[1] > 1:
            raise TypeError("Output must be 1d.")
        else:
            model = res['model']
            rho = res['rho']
            n_ext = min(df_tt[model[0]].shape[0], res['sample_rate'])
            if n_ext < res['sample_rate']:
                print('Insufficient exploitation samples; {} more are needed'.format(res['sample_rate']-n_ext))
            df_ttt = df_tt[model[0]][0:n_ext,:]
            if len(model)>1:
                for i in range(1, len(model)):
                    df_ttt = np.concatenate((df_ttt, df_tt[model[i]][0:n_ext,:]), axis=1)
            df = add_intercept(df_ttt)
            temp = df@res['coef']
            emulator = MvECDF(temp)
            def cdf_cv(x):
                return max(min(emulator.ecdf_eval(x)*rho(x)[1] + rho(x)[0],1),0)
            x_discrete = np.sort(np.concatenate([res['hf_data'].flatten(),
                                                 res['lf_data'].flatten(),
                                                 temp.flatten()]))
            y_discrete = np.sort(np.array([cdf_cv(x) for x in x_discrete]))
            x_min, x_max = min(x_discrete), max(x_discrete)
            y_min, y_max = min(y_discrete), max(y_discrete)
            cdf_temp = interp1d(x_discrete, y_discrete)
            cdf_temp_inv = interp1d(y_discrete, x_discrete)
            def cdf(x):
                if x<=x_min:
                    return 0
                elif x>=x_max:
                    return 1
                else:
                    return cdf_temp(x)
            def quantile(x):
                if x<=y_min:
                    return x_min
                elif x>=y_max:
                    return x_max
                else:
                    return cdf_temp_inv(x)
            if inverse == False:
                return cdf
            else:
                return {'cdf': cdf, 'quantile': quantile}
    