#######################################################################################
# Expectation-Maximization Algorithm for Clustering
#######################################################################################
import gc
import time
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt


def loop_condition(new_mu_list, old_mu_list, epsilon):
    s = 0
    for i in range(len(new_mu_list)):
        si = 0
        for j in range(len(new_mu_list[0])):
            si += (new_mu_list[i][j] - old_mu_list[i][j]) ** 2
        s += si
    if s > epsilon:
        return True
    return False


def mvn_density(x, mu, sigma):
    """
    x = X_j
    mu = mu_i
    sigma = sigma_i
    """
    assert type(x) is np.ndarray
    assert type(mu) is np.ndarray
    d = len(x)
    first_part = 1 / ((np.pi ** (d / 2)) * ((np.linalg.det(sigma)) ** 0.5))
    x_mu = x - mu
    exp_part = np.exp(-0.5 * np.dot(np.dot(x_mu.T, np.linalg.inv(sigma)), x_mu))
    return first_part * exp_part


def Gk(data, k=2, epsilon=0.001):
    """
    1. initializing each Gaussian
    2. deciding ties
    3. stopping criteria
    """
    if type(data) is not np.ndarray:
        data = np.array(data)
    # check if y in data
    t = 0
    n, d = len(data), len(data[0])
    ########################################################
    # Initialization
    # μi is randomly selected from D for each cluster.
    new_mu_list = [data[j] for j in np.random.choice(range(len(data)), k, replace=False)]

    # For each cluster, the covariance matrix is a d × d identity matrix.
    new_sigma_list = [np.identity(d) for i in range(k)]

    # The priors are uniformly initialized
    new_prior_list = [1 / k for i in range(k)]
    mu_list = None
    #########################################################
    # Repeat
    while (t == 0) or (loop_condition(new_mu_list, mu_list, epsilon)):
        t += 1
        mu_list = new_mu_list
        sigma_list = new_sigma_list
        prior_list = new_prior_list

        # Expectation Step
        w = [0 for i in range(k)]
        for i in range(k):
            w[i] = [0 for j in range(n)]
        # posterior probability
        for j in range(n):
            sum_f = 0
            f_xj = []
            for i in range(k):
                f_xj_i = mvn_density(x=data[j], mu=mu_list[i], sigma=sigma_list[i])
                sum_f += f_xj_i * prior_list[i]
                f_xj.append(f_xj_i * prior_list[i])
            for i in range(k):
                w[i][j] = f_xj[i] / sum_f

        # Maximization Step
        new_mu_list, new_sigma_list, new_prior_list = [], [], []
        for i in range(k):
            # re-estimate mean, covariance matrix, priors
            wi_sumj = np.sum([w[i][j] for j in range(n)])
            new_mu_list_numerator = np.array([0 for di in range(d)])
            for j in range(n):
                new_mu_list_numerator = new_mu_list_numerator + w[i][j] * data[j]
            new_mu_list.append(new_mu_list_numerator / wi_sumj)
            new_sigma_list_numerator = np.array([[0 for dj in range(d)] for di in range(d)])
            for j in range(n):
                new_sigma_list_numerator = new_sigma_list_numerator + w[i][j] * np.dot((data[j] - mu_list[i]).reshape(d, 1), (data[j] - mu_list[i]).reshape(1, d))
            new_sigma_list.append(new_sigma_list_numerator / wi_sumj)
            new_prior_list.append(wi_sumj / n)
    return new_mu_list, new_sigma_list, new_prior_list, t
 
 
 ###########################################################################################
 # Analysis of the EM over Real-world Data Sets
 ###########################################################################################
 # Define Error Rate
 ###########################################################################################
 def error_rate(predict_y, true_label):
    label_set = set(true_label)
    if type(true_label[0]) is not int:
        trans_dict = {}
        i = 0
        for c in label_set:
            trans_dict[c] = i
            i += 1
        true_label = [trans_dict[k] for k in true_label]
        label_set = set(true_label)
    predict_label_set = set(predict_y)
    # We need to consider the cases that the cluster number k we assume is larger than real k (==2)
    # So we use the most common 2 clusters when we are calculating error rate
    trans_dict1, trans_dict2 = {}, {}
    a_counter = collections.Counter(predict_y).most_common()
    a_counter = [cas[0] for cas in a_counter]
    most_common_two = a_counter[:2]
    a_counter2 = [most_common_two[1], most_common_two[0]] + a_counter[2:]
    i = 0
    for c in a_counter:
        trans_dict1[c] = i
        i += 1
    i = 0
    for c in a_counter2:
        trans_dict2[c] = i
        i += 1

    predict_y1 = [trans_dict1[k] for k in predict_y]
    predict_y2 = [trans_dict2[k] for k in predict_y]

    err_dict1 = {i: {c: 0 for c in ['g', 'b']} for i in label_set}
    err_dict2 = {i: {c: 0 for c in ['g', 'b']} for i in label_set}
    for j in range(len(predict_y)):
        if true_label[j] == 0:
            true_c, alter_c = 0, 1
        else:
            true_c, alter_c = 1, 0
        if predict_y1[j] == true_c:
            err_dict1[true_c]['g'] += 1
        else:
            err_dict1[true_c]['b'] += 1
        if predict_y2[j] == true_c:
            err_dict2[true_c]['g'] += 1
        else:
            err_dict2[true_c]['b'] += 1

    for i in err_dict1:
        err_dict1[i]['err_rate'] = err_dict1[i]['b'] / (err_dict1[i]['b'] + err_dict1[i]['g'])
    for i in err_dict2:
        err_dict2[i]['err_rate'] = err_dict2[i]['b'] / (err_dict2[i]['b'] + err_dict2[i]['g'])
    return np.min([np.sum([err_dict1[i]['err_rate'] for i in err_dict1]), np.sum([err_dict2[i]['err_rate'] for i in err_dict2])])


 ###########################################################################################
 # Whisker Plot
 ###########################################################################################
 def predict_label(x, mu_list, sigma_list):
    class_n = len(mu_list)
    candidate_prob = [mvn_density(x, mu_list[i], sigma_list[i]) for i in range(class_n)]
    return np.argmax(candidate_prob)

def load_data(main_dir,  data_name):
    if data_name == 'Dataset':
        raw_data = pd.read_csv('{}{}.data'.format(main_dir, data_name), sep="\s+", header=None)
    else:
        raw_data = pd.read_csv('{}{}.data'.format(main_dir, data_name), header=None)
        # We observe that for column 1, all the values are zeros. So we delete this column
        del raw_data[1]
    label_data = raw_data.iloc[:, -1]
    x_data = raw_data.iloc[:, :-1]
    data = np.array(x_data)
    return data, label_data

def main_process(main_dir='B505_data/', data_name='Dataset', epsilon=0.001):
    """
    :param data_name: Dataset, ionosphere
    :return:
    """
    data, label_data = load_data(main_dir,  data_name)

    data_n = len(data)
    total_record = {}

    for k in range(2, 6):
        print('Number of Clusters (k) = {}'.format(k))
        total_record[k] = {}
        for ri in range(14, 20):
            print('\n{} run..................'.format(ri))
            mu_list_r, sigma_list_r, prior_list_r, iter_r = Gk(data, k=k, epsilon=epsilon)
            predict_y_raw = [predict_label(data[j], mu_list_r, sigma_list_r) for j in range(data_n)]
            time.sleep(5)
            total_error_rate = error_rate(predict_y=predict_y_raw, true_label=label_data.tolist())
            total_record[k][ri] = {'mu': mu_list_r, 'sigma': sigma_list_r, 'prior': prior_list_r, 'iter': iter_r,
                                  'total_error_rate': total_error_rate, 'predict': predict_y_raw}
            np.save('{}_record.npy'.format(data_name), total_record, allow_pickle=True)
            del mu_list_r, sigma_list_r, prior_list_r, iter_r, total_error_rate
            gc.collect()
        np.save('{}_record.npy'.format(data_name), total_record, allow_pickle=True)


def whisket_plot(save_path='', item='iter', k1=2, k2=5, runs=20):
    # 1 - iteration number vs k
    # 2 -error rate vs k
    assert item in ['iter', 'total_error_rate']
    Fig = plt.figure()
    if item == 'iter':
        notation_text = 'iteration counts'
    else:
        notation_text = 'error rate'
    for data_name, box_c in zip(['Dataset', 'ionosphere'], ['orange', 'blue']):
        total_record = np.load('{}_record.npy'.format(data_name), allow_pickle=True).item()
        df = pd.DataFrame(index=[str(ki) for ki in range(k1, k2+1)], columns=[t for t in range(runs)])
        for k in range(k1, k2 + 1):
            record_k = [total_record[k][t][item] for t in range(runs)]
            record_k.sort()
            df.loc[str(k)] = record_k
            # for t in range(runs):
            #     df.loc[k, t] = total_record[k][t][item]
        # df.T.boxplot(vert=False)
        box1 = plt.boxplot(df.T, vert=False, labels=[2,3,4,5], patch_artist=True)
        for item2 in ['boxes', 'whiskers', 'fliers', 'caps']:
            plt.setp(box1[item2], color=box_c)
    plt.title('Whisker Plot of Ringnorm(orange) and Ionosphere(blue) Data Set')
    plt.xlabel(notation_text)
    plt.ylabel('Number of clusters (k)')
    plt.savefig('{}{}-whisker.png'.format(save_path, item))
    plt.show()
    
    
#######################################################################
if __name__ == '__main__':
    for data_name in ['Dataset1', 'Dataset2']:
        main_process(main_dir='', data_name=data_name, epsilon=0.001)
    for whisker_x in ['iter', 'total_error_rate']:
        whisket_plot(save_path='', item=whisker_x, k1=2, k2=5, runs=20)
