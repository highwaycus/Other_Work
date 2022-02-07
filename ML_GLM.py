'''
GLM  for Logistic regression, Poisson regression and Ordinal regression. 
Use one generic implementation for the main algorithm that works for multiple observation likelihoods.
'''
import csv
import numpy as np
import math
import time
import matplotlib.pyplot as plt


##########################################
# Data Processing
##########################################
def combine_csv_detail(save_path, file):
    data = []
    with open(save_path + file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            row1 = row[0].split(',')
            row1 = [float(r) for r in row1]
            data.append(row1)
    return data


def load_csv_file(save_path, file_name):
    """
    Open csv file, restore as list format with len(data)  = N = number of sample points,
    len(data[i]) = number of features for i in 1,..., N
    :return: [row_1, row_2, ..., row_N]
    """
    if not file_name.endswith('csv'):
        file_name += '.csv'
    xdata = combine_csv_detail(save_path, file_name)
    # Add Intercept term
    for i in range(len(xdata)):
        xdata[i].append(1)
    ydata = combine_csv_detail(save_path, 'labels-{}'.format(file_name))
    return xdata, ydata


##########################################
# Calculating GLM
##########################################
def w_distance_pct(w_new, w, p):
    if len(w) != p:
        print('\nlen(w != p')
    ans = np.sum([(w_new[i][0] - w[i][0]) ** 2 for i in range(p)])
    ans = ans / np.sum([(w[i][0]) ** 2 for i in range(p)])
    return ans


def iteration_condition(w_new, w, p, roundi, max_round=100, w_change=0.001):
    if (w_distance_pct(w_new, w, p) < w_change) or (roundi >= max_round):
        return False
    return True


def newton_method(r, phi_x, d, alpha, w, p):
    R = np.diag(r)
    # w <- w - H^(-1)g, g=LogL', H = LogL''
    g = np.dot(phi_x.T, d) - alpha * w
    H = -alpha * np.array([[int(j == i) for j in range(p)] for i in range(p)]) - np.dot(np.dot(phi_x.T, R), phi_x)
    H_inv = np.linalg.inv(H)
    w_new = w - np.dot(H_inv, g)
    return w_new


def create_glm(x, t, distribution='logistic', w_change=0.001):
    assert distribution in ['logistic', 'poisson', 'ordinal']
    n = len(x)
    p = len(x[0])
    alpha = 10
    phi_x = np.array(x)
    round_num = 0
    w_init = [0] * p
    w = np.array([[c] for c in w_init])
    w_new = [[10]] * p
    start_time = time.time()
    likelihoodtime = 0
    ########################################
    if distribution == 'logistic':
        # yi = sigmoid(wT phi(xi))
        while iteration_condition(w_new, w, p, round_num, max_round=100, w_change=w_change):
            # print(np.abs(w_new[0] - w[0]) [0])
            if round_num > 0:
                w = w_new
            round_num += 1
            # w0 = [0 for i in range(n)]
            lltime0 = time.time()
            y_hat = [[sigmoid_func(np.dot(phi_x[i], w)[0])] for i in range(n)]
            # print(y_hat[0])
            y_hat = np.array(y_hat)
            likelihoodtime =  likelihoodtime + time.time() - lltime0
            # first derivative
            d = np.array(t) - y_hat
            # second derivative
            r = np.array([(y_hat[i] * (1 - y_hat[i]))[0] for i in range(n)])
            w_new = newton_method(r, phi_x, d, alpha, w, p)
    #######################################
    elif distribution == 'poisson':
        while iteration_condition(w_new, w, p, round_num, max_round=100, w_change=w_change):
            if round_num > 0:
                w = w_new
            round_num += 1
            lltime0 = time.time()
            a = np.dot(phi_x, w)
            y_hat = [[np.exp(a[i][0])] for i in range(n)]
            y_hat = np.array(y_hat)
            likelihoodtime = likelihoodtime + time.time() - lltime0
            # first derivative
            d = np.array(t) - y_hat
            g = np.dot(phi_x.T, d) - alpha * w
            # second derivative
            r = [c[0] for c in y_hat]
            w_new = newton_method(r, phi_x, d, alpha, w, p)
    #######################################
    elif distribution == 'ordinal':
        k, s = 5, 1
        phi_list = [-np.inf, -2, -1, 0, 1, np.inf]
        while iteration_condition(w_new, w, p, round_num, max_round=100, w_change=w_change):
            # print(np.abs(w_new[0] - w[0]) [0])
            if round_num > 0:
                w = w_new
            round_num += 1
            lltime0 = time.time()
            a = np.dot(phi_x, w)
            y_hat = [[s * sigmoid_func(phi_list[j] - a[i][0]) for j in range(len(phi_list))] for i in range(n)]
            likelihoodtime =  likelihoodtime + time.time() - lltime0
            # first derivative
            d = np.array([[y_hat[i][int(t[i][0])] + y_hat[i][int(t[i][0] - 1)] - 1] for i in range(n)])
            g = np.dot(phi_x.T, d) - alpha * w
            # second derivative
            r = np.array([(s ** 2) * (
                        y_hat[i][int(t[i][0])] * (1 - y_hat[i][int(t[i][0])]) + y_hat[i][int(t[i][0] - 1)] * (
                            1 - y_hat[i][int(t[i][0] - 1)])) for i in range(n)])
            w_new = newton_method(r, phi_x, d, alpha, w, p)
    # print('round_num=', round_num)
    run_time = time.time() - start_time
    return w_new, run_time, round_num, likelihoodtime


def sigmoid_func(zi):
    return 1/(1 + np.exp(-zi))


##########################################
# Evaluation
##########################################
def prediction_test(w_new, test_x, test_t, distribution='logistic'):
    start_t = time.time()
    test_n = len(test_x)
    phi_x = np.array(test_x)
    if distribution == 'logistic':
        y_hat = [[sigmoid_func(np.dot(phi_x[i], w_new)[0])] for i in range(test_n)]
        t_hat = [[int(c[0] > 0.5)] for c in y_hat]
        err = [int(t_hat[i][0] != test_t[i][0]) for i in range(test_n)]
    elif distribution == 'poisson':
        a = np.dot(phi_x, w_new)
        y_hat = [np.exp(a[i][0]) for i in range(test_n)]
        t_hat = [math.floor(y_hat[i]) for i in range(test_n)]
        err = [np.abs(t_hat[i] - test_t[i][0]) for i in range(test_n)]

    elif distribution == 'ordinal':
        k, s = 5, 1
        phi_list = [-np.inf, -2, -1, 0, 1, np.inf]
        a = np.dot(phi_x, w_new)
        y_hat = [[s * sigmoid_func(phi_list[j] - a[i][0]) for j in range(len(phi_list))] for i in range(test_n)]
        p_hat = [[y_hat[i][j] - y_hat[i][j - 1] for j in range(k + 1)] for i in range(test_n)]
        t_hat = [np.argmax(p_hat[i]) for i in range(test_n)]
        err = [np.abs(t_hat[i] - test_t[i][0]) for i in range(test_n)]
    mean_err = np.mean(err)
    classify_time = time.time() - start_t
    return mean_err, classify_time


def main_calculate(save_path, repeat_n=30, file_name_list=None, distribution_assigned=None):
    if not file_name_list:
        file_name_list = ['A', 'usps', 'AP', 'AO']
    record = {}
    portion_list = np.linspace(0.1, 1, 10)
    for file_name in file_name_list:
        print('############################')
        print('Start', file_name, '.........')
        if file_name in ['A', 'usps']:
            distribution = 'logistic'
        elif file_name == 'AP':
            distribution = 'poisson'
        elif file_name == 'AO':
            distribution = 'ordinal'
        else:
            distribution = distribution_assigned
        x_data, y_data = load_csv_file(save_path=save_path, file_name=file_name)
        total_n = len(x_data)
        record[file_name] = {por: {'mean_err': [], 'run_time': [], 'round_n': [], 'cls_t':[], 'likelihood_time':[]} for por in portion_list}
        for rep in range(repeat_n):
            print('rep=', rep)
            test_index = np.random.choice(total_n, size=total_n // 3, replace=False)
            train_index = [i for i in range(total_n) if i not in test_index]
            test_set_x = [x_data[j] for j in test_index]
            test_set_y = [y_data[j] for j in test_index]
            train_set_x = [x_data[j] for j in train_index]
            train_set_y = [y_data[j] for j in train_index]
            train_n = len(train_set_x)
            for portion in portion_list:
                # for portion in [0.1]:
                apply_train_n = int(np.round(portion * train_n))
                apply_train_x = train_set_x[:apply_train_n]
                apply_train_y = train_set_y[:apply_train_n]
                w_train, run_time, round_n, likelihoodtime = create_glm(x=apply_train_x, t=apply_train_y, distribution=distribution)
                mean_err, cls_t = prediction_test(w_new=w_train, test_x=test_set_x, test_t=test_set_y,
                                                  distribution=distribution)
                record[file_name][portion]['mean_err'].append(mean_err)
                record[file_name][portion]['run_time'].append(run_time)
                record[file_name][portion]['round_n'].append(round_n)
                record[file_name][portion]['cls_t'].append(cls_t)
                record[file_name][portion]['likelihood_time'].append(likelihoodtime)
                # print(portion, apply_train_n, w_train[0], mean_err)
        for portion in portion_list:
            record[file_name][portion]['std_dev_mean_error'] = np.std(record[file_name][portion]['mean_err'])
            for item in ['mean_err', 'run_time', 'round_n', 'cls_t', 'likelihood_time']:
                record[file_name][portion]['mean_{}'.format(item)] = np.mean(record[file_name][portion][item])
    np.save(save_path + 'pp3_record.npy', record, allow_pickle=True)
    return record


def plot_learning_curve(sub_record, file_name, save_path):
    plt.clf()
    plt.close()
    x = list(sub_record.keys())
    plt.plot(x, [sub_record[p]['mean_mean_err'] for p in x])
    plt.errorbar(x, [sub_record[p]['mean_mean_err'] for p in x], yerr=[sub_record[p]['std_dev_mean_error'] for p in x], fmt='o', ecolor='r', color='b')
    plt.xlabel('Portion')
    plt.ylabel('mean(error)')
    plt.title(file_name + ': Error' )
    plt.legend()
    plt.savefig(save_path + file_name.replace('.csv', '') + '_error_chart.png')
    # plt.show()
    plt.clf()
    plt.close()
    plt.plot(x, [sub_record[p]['mean_run_time'] for p in x])
    plt.xlabel('Portion')
    plt.ylabel('mean Run Time (secs)')
    plt.title('{}: Run Time'.format(file_name))
    plt.savefig(save_path + file_name.replace('.csv', '') + '_runtime_chart.png')


############################################
def main(save_path):
    file_name_list = ['A', 'usps', 'AP', 'AO']
    record = main_calculate(save_path=save_path, repeat_n=30, file_name_list=file_name_list)
    for file_name in file_name_list:
        plot_learning_curve(record[file_name], file_name, save_path)
    for file_name in file_name_list:
        print(file_name)
        print('mean error=', [np.round(record[file_name][p]['mean_mean_err'], 3) for p in record[file_name]])
        print('stddev error=', [np.round(record[file_name][p]['std_dev_mean_error'], 4) for p in record[file_name]])
        print('mean classify time=', [np.round(record[file_name][p]['mean_cls_t'], 4) for p in record[file_name]])
        print('mean likelihood time=', [np.round(record[file_name][p]['mean_likelihood_time'], 4) for p in record[file_name]])
        print('round_n =', [np.round(record[file_name][p]['mean_round_n'], 4) for p in record[file_name]])
        print('run time per round=', [np.round(record[file_name][p]['mean_run_time'] /record[file_name][p]['mean_round_n'] , 4) for p in record[file_name]])


##########################################################
if __name__ == '__main__':
    main(save_path='data/')
