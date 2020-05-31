from model import *

import cvxpy as cvx
import numpy as np


import pdb


def costFunction(x, y, a, b, p, num=0):
    res = 0
    if num == 0:
        for i in range(0, len(y)):
            res += (p * x[i] + a * np.maximum(0, y[i] - x[i]) + b * np.absolute(x[i] - x[i - 1]))
    else:
        for i in range(0, num):
            res += (p * x[i] + a * np.maximum(0, y[i] - x[i]) + b * np.absolute(x[i] - x[i - 1]))
    return res


def cvx_static_offline_solution(y,a,b,p) :

    def objectiveFunctionStatic(x,y,a,b,p) :
        objective_function = 0
        for i in range(0,len(y)) :
            cost = cvx.multiply(p,x)
            if i == 0 :
                penalty = 0
                #switching_cost = cvx.multiply(a, cvx.maximum(0,x[i]) )
            else :
                penalty =  cvx.multiply(a,   cvx.maximum(0,( y[i] - x ))  )
                #switching_cost =  cvx.multiply( b, cvx.abs(x[i] - x[i-1]) )
            objective_function += cost + penalty #+ switching_cost

        return objective_function

    y_true = y
    a = cvx.Parameter(nonneg=True, value=a)
    b = cvx.Parameter(nonneg=True, value=b)
    x = cvx.Variable() #len(y_true)
    p_cost = cvx.Parameter(nonneg=True, value=p)
    objective_function_minimizer = cvx.Minimize(objectiveFunctionStatic(x,y_true,a,b,p_cost))
    problem = cvx.Problem(objective_function_minimizer, [x >= 0])
    problem.solve(solver=cvx.SCS)
    return [float(x.value)]*len(y)



def cvx_dynamic_offline_solution(y,a,b,p) :

    def objectiveFunctionDynamic(x,y,a,b,p) :
        objective_function = 0
        for i in range(0,len(y)) :
            cost = cvx.multiply(p,x[i])
            penalty =  cvx.multiply(a,   cvx.maximum(0,( y[i] - x[i] ))  )
            switching_cost =  cvx.multiply( b, cvx.abs(x[i] - x[i-1]) )

            objective_function += cost + penalty + switching_cost

        return objective_function

    y_true = y
    a = cvx.Parameter(nonneg=True, value=a)
    b = cvx.Parameter(nonneg=True, value=b)
    x = cvx.Variable(len(y_true)) #
    p_cost = cvx.Parameter(nonneg=True, value=p)
    objective_function_minimizer = cvx.Minimize(objectiveFunctionDynamic(x,y_true,a,b,p_cost))
    problem = cvx.Problem(objective_function_minimizer, [x >= 0])
    problem.solve(solver=cvx.SCS)
    provisions = list(x.value)
    return provisions

def online_gradient_descent(y,a,b,p,step_size_modifier = 1):
    y_true = y

    provisions = [0]*len(y_true)
    for t in range(len(y_true) - 1):
        if t>0:
            step_size =  1/np.sqrt(t)
            descent = 0
            if y_true[t] > provisions[t] :
                if provisions[t] <= provisions[t-1] :
                    descent += (a+b)
                else :
                    descent  = descent - a
            else :
                if provisions[t] <= provisions[t-1] :
                    descent = descent - a
                else :
                    descent = descent - a - b
            provisions[t+1] = provisions[t] + (step_size*descent)

    return [provisions[len(provisions)-1]]*len(y)


def online_balanced_descent(y,a,b,p,step_size_modifier = 1):
    y_true = y

    provisions = [0]*len(y_true)
    for t in range(len(y_true) - 1):
        if t>0:
            step_size =  1/np.sqrt(t)
            descent = 0
            if y_true[t] > provisions[t] :
                if provisions[t] <= provisions[t-1] :
                    descent += (a+b)
                else :
                    descent  = descent - a
            else :
                if provisions[t] <= provisions[t-1] :
                    descent = descent - a
                else :
                    descent = descent - a - b
            provisions[t+1] = provisions[t] + (step_size*descent)

    return [provisions[len(provisions)-1]]*len(y)


def receding_horizon_control(y,a,b,p,window_size):


    y_pred = y
    provisions = [0]*len(y_pred)

    def objective_function(p, a, b,   provision_window, demand_window):
        obj = 0
        for w in range(0, len(demand_window)):
            obj += p * provision_window[w]
            if w == 0:
                obj += cvx.multiply(a, provision_window[w])  # No switching cost
            else:
                obj += cvx.multiply(a, cvx.maximum(0, (demand_window[w] - provision_window[w - 1])))   + cvx.multiply( b, cvx.abs( (provision_window[w] - provision_window[w-1]) ) )
        return obj

    for t in range(0, len(y_pred)):

        demand_window = y_pred[t:t + window_size]
        provision_window = cvx.Variable(window_size)
        obj = objective_function(  p, a, b, provision_window, demand_window)
        problem = cvx.Problem(cvx.Minimize(obj), [provision_window >= 0])
        problem.solve(solver=cvx.SCS)
        provisions[t] = provision_window.value[0]

    return provisions



def commitment_horizon_control(y,a,b,p,window_size,commitment_level=1):
    y_pred = y
    provisions = [0]*len(y_pred)


    def objective_function(p, a, b,   provision_window, demand_window):
        obj = 0
        for w in range(0, len(demand_window)):
            obj += p * provision_window[w]
            if w == 0:
                obj += cvx.multiply(a, provision_window[w])  # No switching cost
            else:
                obj += cvx.multiply(a, cvx.maximum(0, (demand_window[w] - provision_window[w - 1])))   + cvx.multiply( b, cvx.abs( (provision_window[w] - provision_window[w-1]) ) )
        return obj
    all_provisions = [[0]*window_size]*len(y_pred)
    for t in range(0, len(y_pred)):
        try :
            demand_window = y_pred[t:t + window_size]
            provision_window = cvx.Variable(window_size)
            obj = objective_function(  p, a, b, provision_window, demand_window)
            problem = cvx.Problem(cvx.Minimize(obj), [provision_window >= 0])
            problem.solve(solver=cvx.SCS)
            all_provisions[t] = list(provision_window.value)
        except Exception as e :
            print(e)
    for t in range(0,len(y_pred)) :
            t_sum = 0
            timestep = t + 1

            if timestep <= commitment_level :
                start = 0
            else :
                start = timestep - commitment_level

            for x in range(start,start+commitment_level) :
                y = t-x
                if y < 0 :
                    y = 0
                try :
                    t_sum += all_provisions[x][y]
                except Exception as e :
                    print(e)
            provisions[t] = t_sum/commitment_level if timestep > commitment_level else t_sum/timestep

    return provisions


def fetch_weights(y_true,y_pred,a,b,p,window_size=5,commitment_level=3):
    algorithm_weights = {
        "ogd" : .25,
        "dos" : .25,
        "rhc" : .25,
        "chc" : .25
    }
    provisions = {
        "ogd" : [0]*672,
        "dos" : [0]*672,
        "rhc" : [0]*672,
        "chc" : [0]*672
    }

    for t in range(0,len(y_pred)) :
        x = {}
        if t%10 == 0 :
            print("Processed {0} of {1} items".format(t,len(y_pred)))
        x["ogd"] = online_gradient_descent(y_pred[0:window_size],a,b,p)[0]
        x["dos"] = cvx_dynamic_offline_solution(y_pred[0:window_size],a,b,p)[0]
        x["rhc"] = receding_horizon_control(y_pred[0:window_size],a,b,p,window_size)[0]
        timestep = t+1
        buffer = window_size - commitment_level
        if timestep < commitment_level :
            prediction_window = y_pred[t:t+window_size]
        else :
            prediction_window = y_pred[ t-(window_size-commitment_level):t+window_size  ]
        x["chc"] = commitment_horizon_control(prediction_window,a,b,p,window_size,commitment_level)[0]

        algos = list(algorithm_weights.keys())
        new_algorithm_weights = {}
        for algo in algos :
            if t == 0 :
                prev_provision = 0
            else :
                prev_provision = provisions[algo][t-1]
            cur_cost = (p * x[algo] + a * np.maximum(0, y_true[t] - x[algo]) + b * np.absolute(x[algo] - prev_provision))
            sum_cost = 0
            for algo2 in algos :
                if t == 0:
                    prev_provision = 0
                else:
                    prev_provision = provisions[algo2][t - 1]
                sum_cost += (p * x[algo2] + a * np.maximum(0, y_true[t] - x[algo2]) + b * np.absolute(x[algo2] - prev_provision))

            new_algorithm_weights[algo] = (1/np.sqrt(timestep))*((1/cur_cost)/(1/sum_cost)) + (1 - (1/np.sqrt(timestep)) )*(algorithm_weights[algo])
        algorithm_weights = new_algorithm_weights
        provisions["ogd"][t] = x["ogd"]
        provisions["dos"][t] = x["dos"]
        provisions["rhc"][t] = x["rhc"]
        provisions["chc"][t] = x["chc"]

    return algorithm_weights

def randomized_algorithm_provisioning(y_true,y_pred,a,b,p,window_size,commitment_level,weights):
    x= {}
    x["ogd"] = online_gradient_descent(y_true,a,b,p)
    x["dos"] = cvx_dynamic_offline_solution(y_true,a,b,p)
    x["rhc"] = receding_horizon_control(y_pred,a,b,p,window_size)
    x["chc"] = commitment_horizon_control(y_pred,a,b,p,commitment_level)

    pr_weights = {}
    best_weight = 0
    for algo in list(weights.keys()) :
        pr_weights[algo] = weights[algo]/sum([ weights[algo2] for algo2 in weights.keys() ])
        if pr_weights[algo] > best_weight:
            best_weight = pr_weights[algo]
    final_provisions = [0]*len(y_true)
    for t in range(0,len(y_true)) :
        prv = 0
        for algo in list(x.keys()) :
            prv = prv + (best_weight/sum(list(weights.values())))*x[algo][t]
        final_provisions[t] = prv
    return final_provisions

def deterministic_algorithm_provisioning(y_true,y_pred,a,b,p,window_size,commitment_level,weights) :
    x = {}
    x["ogd"] = online_gradient_descent(y_true, a, b, p)
    x["dos"] = cvx_dynamic_offline_solution(y_true, a, b, p)
    x["rhc"] = receding_horizon_control(y_pred, a, b, p, window_size)
    x["chc"] = commitment_horizon_control(y_pred, a, b, p, commitment_level)



    final_provisions = [0]*len(y_true)
    for t in range(0, len(y_true)):
        prv = 0
        for algo in list(x.keys()):
            prv = prv + ( weights[algo]/sum(list(weights.values()))  * x[algo][t])
        final_provisions[t] = prv

    return final_provisions

if __name__ == "__main__" :






    '''
    randomized_cost = {}
    deterministic_cost = {}
    for house in ["b", "c", "f"]: #

       if house == "b":
           y_true = pd.read_csv(house + "_y_true.csv", index_col="time")['2014-11-01 00:00:00':'2014-11-14 23:30:00']
       if house == "c":
           y_true = pd.read_csv(house + "_y_true.csv", index_col="time")['2015-11-01 00:00:00':'2015-11-14 23:30:00']
       if house == "f":
           y_true = pd.read_csv(house + "_y_true.csv", index_col="time")['2016-11-01 00:00:00':'2016-11-14 23:30:00']
       y_true = list(y_true['total_energy'].values)

       for model in ["linear"]:
            if house == "b":
               y_pred = pd.read_csv(house + "_" + model + "_y_pred.csv", index_col="time")[
                        '2014-11-01 00:00:00':'2014-11-14 23:30:00']
            if house == "c":
               y_pred = pd.read_csv(house + "_" + model + "_y_pred.csv", index_col="time")[
                        '2015-11-01 00:00:00':'2015-11-14 23:30:00']
            if house == "f":
               y_pred = pd.read_csv(house + "_" + model + "_y_pred.csv", index_col="time")[
                        '2016-11-01 00:00:00':'2016-11-14 23:30:00']

            y_pred = list(y_pred['predictions'].values)

            #weights = {'ogd': 1.8566169862700934, 'dos': 12.507848396733426, 'rhc': 12.38946409283843, 'chc': 4.0230959508790685}
            weights = fetch_weights(y_true, y_pred, 4, 4, .4, 5, 3)
            x = deterministic_algorithm_provisioning(y_true,y_pred,4,4,.4,5,3,weights)
            deterministic_cost[house] = [costFunction(x,y_true,4,4,.4)]
            x = randomized_algorithm_provisioning(y_true,y_pred,4,4,.4,5,3,weights)
            randomized_cost[house] = [costFunction(x,y_true,4,4,.4)]
    try :
        df = pd.DataFrame(randomized_cost)
        df.to_csv("randomized_cost.csv")

        df = pd.DataFrame(deterministic_cost)
        df.to_csv("deterministic_cost.csv")
    except Exception as e :
        pdb.set_trace()




   best_cost = []
   for house in ["b", "c", "f"]:

       if house == "b":
           y_true = pd.read_csv(house + "_y_true.csv", index_col="time")['2014-11-01 00:00:00':'2014-11-14 23:30:00']
       if house == "c":
           y_true = pd.read_csv(house + "_y_true.csv", index_col="time")['2015-11-01 00:00:00':'2015-11-14 23:30:00']
       if house == "f":
           y_true = pd.read_csv(house + "_y_true.csv", index_col="time")['2016-11-01 00:00:00':'2016-11-14 23:30:00']
       y_true = list(y_true['total_energy'].values)

       for model in ["linear"]:
           if house == "b":
               y_pred = pd.read_csv(house + "_" + model + "_y_pred.csv", index_col="time")[
                        '2014-11-01 00:00:00':'2014-11-14 23:30:00']
           if house == "c":
               y_pred = pd.read_csv(house + "_" + model + "_y_pred.csv", index_col="time")[
                        '2015-11-01 00:00:00':'2015-11-14 23:30:00']
           if house == "f":
               y_pred = pd.read_csv(house + "_" + model + "_y_pred.csv", index_col="time")[
                        '2016-11-01 00:00:00':'2016-11-14 23:30:00']

           y_pred = list(y_pred['predictions'].values)

           for ab in [[2,2],[1,1],[4,2],[4,1],[1,4],[2,4]]:
               print("Performing CHC for house " + house + " for model " + model + " for [a,b] " + str(ab))
               x = commitment_horizon_control(y_pred, ab[0], ab[1], .4, 9, 7)
               cost = costFunction(x, y_true, ab[0], ab[1], .4)
               best_cost.append({"house": house, "model": model, "cost": cost, "commitement": 7, "window": 9,"a":ab[0],"b":ab[1]})

       df = pd.DataFrame(best_cost)
       df.to_csv("best_model_cost.csv")
  '''
commitment_cost = []
for house in ["b","c","f"] :

  if house == "b" :
     y_true_1 = pd.read_csv(house + "_y_true.csv",index_col="time")['2014-11-01 00:00:00':'2014-11-14 23:30:00']
  if house == "c" :
      y_true_1 = pd.read_csv(house + "_y_true.csv", index_col="time")['2015-11-01 00:00:00':'2015-11-14 23:30:00']
  if house == "f" :
      y_true_1 = pd.read_csv(house + "_y_true.csv", index_col="time")['2016-11-01 00:00:00':'2016-11-14 23:30:00']
  y_true = list(y_true_1['total_energy'].values)



  for model in ["linear"] : #"xgb","lstm",
    if house == "b" :
     y_pred_1 = pd.read_csv(house+"_"+model+"_y_pred.csv",index_col="time")['2014-11-01 00:00:00':'2014-11-14 23:30:00']
    if house == "c" :
      y_pred_1 = pd.read_csv(house+"_"+model+"_y_pred.csv", index_col="time")['2015-11-01 00:00:00':'2015-11-14 23:30:00']
    if house == "f" :
      y_pred_1 = pd.read_csv(house+"_"+model+"_y_pred.csv", index_col="time")['2016-11-01 00:00:00':'2016-11-14 23:30:00']

    y_pred = list(y_pred_1['predictions'].values)

    weights = weights = {'ogd': 1.8566169862700934, 'dos': 12.507848396733426, 'rhc': 12.38946409283843, 'chc': 4.0230959508790685}
    x1 = deterministic_algorithm_provisioning(y_true, y_pred, 4, 4, .4, 5, 3, weights)
    #deterministic_cost[house] = [costFunction(x1, y_true, 4, 4, .4)]
    x2 = randomized_algorithm_provisioning(y_true, y_pred, 4, 4, .4, 5, 3, weights)
    #randomized_cost[house] = [costFunction(x2, y_true, 4, 4, .4)]

    y_pred_1['randomized provisioning'] = x2
    y_pred_1['deterministic provisioning'] = x1

    y_pred_1.plot(figsize=(10, 5), grid=True)
    plt.xlabel("TIME")
    plt.ylabel("Load in KW")
    plt.title("Predicted vs Provisioned load for home "+house+" using Commitement Horizon Control (Linear Regression)")
    plt.show()
    y_pred_1 = y_pred_1.drop(columns=['randomized provisioning','deterministic provisioning'])

  #df = pd.DataFrame(commitment_cost)
  #df.to_csv("commitment_cost.csv")
