import torch
from osrm import TSP_route
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
import numpy as np
import random
from Worker import accept_rate, norm


class Platform():
    def __init__(self,discount_factor=0.99, njobs=24, probability_worker = False):
        super().__init__()
        self.reset(discount_factor)
        self.njobs = njobs
        self.probability_worker = probability_worker

    def reset(self,discount_factor=0.99):
        self.discount_factor = discount_factor
        self.Total_Reward = 0
        # self.Total_Detour = []
        self.PickUp = 0
        self.worker_reject = 0

        self.direct_time = []
        self.workload = []
        # self.detour_time = [] # work_time - direct_time
        self.valid_distance = []
        self.price_sigma_pos = []
        self.price_sigma_neg = []
        self.price_mu_pos = []
        self.price_mu_neg = []
        self.price_pos = []
        self.price_neg = []


    def assign(self,q_matrix):
        # TODO: set non-zero threshold
        threshold = 0
        # Solve Bipartite Match Process with ILP
        num_vehicles, num_demands = q_matrix.shape
        Value_Matrix = np.concatenate((q_matrix,np.zeros_like(q_matrix)+threshold),axis=1)
        cost_matrix = -Value_Matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # 创建一个列表来保存每个车辆被分配的订单
        assignment = [None] * len(Value_Matrix)

        # 获取每个车辆被分配的订单
        for i in range(len(row_indices)):
            if col_indices[i] >= num_demands:
                assignment[row_indices[i]] = None
            else:
                assignment[row_indices[i]] = col_indices[i]

        # 计算最大化的值
        max_value = -1 * cost_matrix[row_indices, col_indices].sum()

        # 返回分配结果和最大值
        return assignment, max_value

    def feedback(self, observe, reservation_value, speed, current_order, current_order_num, assignment, new_orders_state, price_mu, price_sigma, reward_func, punish_rate, threshold, current_time, worker_Q_Net, exploration_rate, worker_reject_punishment, device, worker_state, mask):
        feedback_table = []
        new_route_table = []
        new_route_time_table = []
        new_remaining_time_table = []
        new_total_travel_time_table = []
        accepted_orders = []
        worker_feed_back_table = []

        results = Parallel(n_jobs=self.njobs)(
            delayed(excute)(observe[i], reservation_value[i], speed[i], current_order[i], current_order_num[i], assignment[i], new_orders_state, price_mu[i], price_sigma[i], reward_func, punish_rate, threshold, current_time, worker_Q_Net, exploration_rate, worker_reject_punishment, device, worker_state[i], self.probability_worker, mask[i])
            for i
            in range(observe.shape[0]))

        # for i in range(observe.shape[0]):
        #     excute(observe[i], reservation_value[i], current_order[i], current_order_num[i], assignment[i], new_orders_state, price_mu[i], price_sigma[i], reward_func, punish_rate, threshold, current_time)
        # exit()

        for i in range(len(results)):
            result = results[i]
            feedback_table.append(result[0])
            new_route_table.append(result[1])
            new_route_time_table.append(result[2])
            new_remaining_time_table.append(result[3])
            new_total_travel_time_table.append(result[4])
            # timeout = result[5]
            if assignment[i] is not None and result[0] is not None:
                self.Total_Reward += self.discount_factor ** current_time * result[0][2] # '''result[0][2]: reward'''
                # self.Total_Detour.append(timeout)

            if result[6] is not None:
                accepted_orders.append(result[6])
                self.PickUp += 1
            elif result[0] is not None:
                self.worker_reject += 1
            worker_feed_back_table.append(result[7])

            if result[8] is not None:
                if len(result[8])>2:
                    self.direct_time.append(result[8][0])
                    self.workload.append(result[8][1])
                    self.valid_distance.append(result[8][2])
                    self.price_sigma_pos.append(result[8][3])
                    self.price_mu_pos.append(result[8][4])
                    self.price_pos.append(result[0][1][0])
                else:
                    self.price_sigma_neg.append(result[8][0])
                    self.price_mu_neg.append(result[8][1])
                    self.price_neg.append(result[0][1][0])


        return feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, accepted_orders, worker_feed_back_table

'''
beta_list:
beta_list[0]: reward of taking new order
beta_list[1]: reward of client paying
beta_list[2]: punishment of paying salary
beta_list[3]: punishment of timeout orders
beta_list[4]: punishment of added time
beta_list[5]: add punishment to the over time
beta_list[6]: add punishment to unit price directly
'''
def reward_func_generator(beta_list, threshold):
    def reward(time_add,time_out,salary,direct_distance,unit_price):
        if time_add <= threshold:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * salary / 100 - beta_list[3] * time_out - beta_list[4] * time_add / 60 - beta_list[6] * (unit_price - 1.0)
        else:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * salary / 100 - beta_list[3] * time_out - beta_list[4] * time_add / 60 - beta_list[6] * (unit_price - 1.0) - beta_list[5] * (time_add - threshold) / 60
        return r
    return reward


'''
excute for each worker

input:
observe, reservation_value, current_order, current_order_num: worker information
assignment: assignment which order to this worker
new_orders_state: the state matrix of all new orders
price_mu_vector, price_sigma_vector: the price distribution of each order (of this worker)
reward_func, punish_rate, threshold: to calculate the reward
current_time, worker_Q_Net, exploration_rate, worker_reject_punishment, device, worker_state: used for worker network

output:
feedback: [[observe, current_order, current_order_num, new_orders_state[assignment], time], [price, price_log_prop], reward, pickup_time] -- (s,a,r,pick_t)
new_route, new_route_time: route/time of each nodes which the worker will pass
new_time, new_total_travel_time: remaining/total time of each order
timeout: how many orders will be timeout after this assignment
accept_order: accepted order number
worker_feedback: used for worker network
log: direct transportation time & workload time
'''
def excute(observe, reservation_value, speed, current_order, current_order_num, assignment, new_orders_state, price_mu_vector, price_sigma_vector, reward_func, punish_rate, threshold, current_time, worker_Q_Net, exploration_rate, worker_reject_punishment, device, worker_state, probability_worker, mask):
    if mask == 2:
        return None, None, None, None, None, 0, None, None, None
    current_order_num = int(current_order_num)
    if assignment is not None and observe[3] == 0:
        price_mu, price_sigma = price_mu_vector[assignment], price_sigma_vector[assignment]
        price_dist = torch.distributions.Normal(price_mu, price_sigma)
        price = price_dist.sample()
        if price<0:
            # price=price-price+1e-8
            price = price - price
        elif price>1.2:
            price = price - price + 1.2
        price_log_prop = price_dist.log_prob(price)
        price, price_log_prop = price.item(), price_log_prop.item() # avoid gradient propagation

        if worker_Q_Net is not None: # intelligent worker
            rand = random.random()
            if rand<exploration_rate:
                acc_rate = 0.5
            else:
                # worker_Q_Net.eval()
                torch.set_grad_enabled(False)
                order_norm, x_norm, x_order_norm = norm(np.expand_dims(new_orders_state[assignment], axis=0),
                                                        np.expand_dims(worker_state, axis=0),
                                                        np.expand_dims(current_order, axis=0))
                x = np.concatenate([order_norm, x_norm, np.array([[reservation_value, price]])], axis=-1)
                x = torch.from_numpy(x).to(device)
                x_order_norm = torch.from_numpy(x_order_norm).to(device)
                worker_q_value = worker_Q_Net(x, x_order_norm, torch.tensor([current_order_num]).to(device))
                # acc_rate = int(worker_q_value[0, 1]>=worker_q_value[0, 0])

                acc_rate = torch.softmax(worker_q_value[0], dim=-1)
                acc_rate = acc_rate[1]


            # worker_Q_Net.eval() # no ε-greedy to avoid platform network cannot converge
            # torch.set_grad_enabled(False)
            # order_norm, x_norm, x_order_norm = norm(np.expand_dims(new_orders_state[assignment], axis=0), np.expand_dims(worker_state, axis=0), np.expand_dims(current_order, axis=0))
            # x = np.concatenate([order_norm, x_norm, np.array([[reservation_value, price]])], axis=-1)
            # x = torch.from_numpy(x).to(device)
            # x_order_norm = torch.from_numpy(x_order_norm).to(device)
            # worker_q_value = worker_Q_Net(x, x_order_norm, torch.tensor([current_order_num]).to(device))
            # acc_rate = worker_q_value[0,1]

            rand = random.random()
            flag = (rand<=acc_rate)
        else:
            # print(price,reservation_value,price<reservation_value)
            if probability_worker:
                acc_rate = accept_rate(price,reservation_value)
                rand = random.random()
                flag = (rand<=acc_rate)
            else:
                if price<reservation_value:
                    flag = False
                else:
                    flag = True


        if flag: #accept
            accept_order = assignment
            # 1. direct_distance
            plat, plon, dlat, dlon = new_orders_state[assignment,:4]
            direct_route, direct_route_time, direct_time, direct_distance = TSP_route((plat, plon), [(dlat, dlon)])
            # direct_time = direct_time[0]
            # 2. pickup
            pickup_route, pickup_route_t, pickup_time, _ = TSP_route((observe[0].item(),observe[1].item()), [(plat, plon)])
            # 3. add the new order
            destination_points = []
            for i in range(current_order_num):
                destination_points.append((current_order[i,0].item(),current_order[i,1].item()))
            destination_points.append((dlat, dlon))
            new_route, new_route_time, new_time, _ = TSP_route((plat, plon), destination_points)
            if len(new_route)!=0:
                # 4. calculate reward

                original_total_travel_time = np.sum(current_order[:,3])

                new_total_travel_time = np.array(new_time)
                new_total_travel_time[:-1] = new_total_travel_time[:-1] + current_order[:current_order_num,3] - current_order[:current_order_num,2] # add the time already cost for each old order
                new_total_travel_time = new_total_travel_time + pickup_time[0]

                timeout = np.sum(new_total_travel_time / speed > threshold) # how many orders will be over time
                time_add = np.sum(new_total_travel_time) - original_total_travel_time # total added time of all orders

                # added workload
                work_add = np.max(new_total_travel_time) - np.max(current_order[:, 3])
                salary = work_add * price
                reward = reward_func(time_add / speed, timeout, salary, direct_distance, price)
                feedback = [[observe,current_order,current_order_num,new_orders_state[assignment], current_time], [price,price_log_prop,work_add,salary], reward, pickup_time[0]]

                # worker_reward = work_add * (price-reservation_value)
                # worker_reward = work_add * (price / reservation_value)
                worker_reward = salary

                worker_feedback = [1,worker_reward,price,work_add]

                log = [direct_time, work_add, direct_distance, price_sigma, price_mu]

                return feedback, new_route, new_route_time, new_time, new_total_travel_time, timeout, accept_order, worker_feedback, log
            else: # routing failure --> decline the assignment
                return None, None, None, None, None, 0, None, None, None
        else: #reject
            # reward = - punish_rate / price # to help model increase the price
            reward = - punish_rate
            feedback = [[observe, current_order, current_order_num, new_orders_state[assignment], current_time],
                        [price, price_log_prop], reward, -1]


            worker_reward = -worker_reject_punishment
            # worker_reward = -worker_reject_punishment*price/reservation_value
            worker_feedback = [0, worker_reward, price, 0]
            return feedback, None, None, None, None, 0, None, worker_feedback, [price_sigma, price_mu]
    else:
        return None, None, None, None, None, 0, None, None, None