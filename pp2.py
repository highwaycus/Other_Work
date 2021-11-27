# Machine Learning B555, Fall 2021
# Programming Project 2
# Andrew Huang, UID:2000921832

############################################
# Task 1
import csv
import numpy as np
import matplotlib.pyplot as plt


def open_csv_file(file_name, save_path='pp2/pp2data/pp2data/'):
    """
    Open csv file, restore as list format with len(data)  = N = number of sample points,
    len(data[i]) = number of features for i in 1,..., N
    :return: [row_1, row_2, ..., row_N]
    """
    data = []
    with open(save_path + file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            row1 = row[0].split(',')
            row1 = [float(r) for r in row1]
            data.append(row1)
    return data


def linear_regression_predict(phi, t, lambda_value):
    # linear regression model with regularization
    # return w
    lam_matrix = [[int(j == i) * lambda_value for j in range(len(phi[0]))] for i in range(len(phi[0]))]
    # w = (lambda*I + phi^T phi)^(-1) phi^T t
    ans = (lam_matrix + np.dot(np.transpose(phi), phi))
    ans = np.linalg.inv(ans)
    phi_t = np.dot(np.transpose(phi), t)
    w = np.dot(ans, phi_t)
    return w


def mse_calculate(y_true, y_hat):
    # calculate MSE = sum((yi-y_hat)**2)/n
    n = len(y_true)
    mse = sum([(y_true[i] - y_hat[i]) ** 2 for i in range(n)]) / n
    if type(mse) is np.ndarray:
        mse = mse[0]
    return mse


def mse_main(phi, t, w):
    # predict target and calculate MSE
    y_hat = np.dot(phi, w)
    mse_ans = mse_calculate(t, y_hat)
    return mse_ans


def main_task_1(save_path='pp2/pp2data/pp2data/'):
    group_list = ['crime', 'wine', 'artsmall', 'artlarge']
    lambda_v_list = [v for v in range(1, 151)]
    total_record = {}
    for group in group_list:
        record = {}
        start_time = time.time()
        for lam_v in lambda_v_list:
            # reg is the value of lambda, the regularization parameter
            phi = open_csv_file('train-{}.csv'.format(group))
            t = open_csv_file('trainR-{}.csv'.format(group))
            test_data = open_csv_file('test-{}.csv'.format(group))
            test_data_R = open_csv_file('testR-{}.csv'.format(group))
            w = linear_regression_predict(phi, t, lam_v)
            mse_train = mse_main(phi, t, w)
            mse_test = mse_main(test_data, test_data_R, w)
            record[lam_v] = {'train_mse': mse_train, 'test_mse': mse_test}
        # plot mse
        Figure = plt.figure()
        plt.scatter(x=lambda_v_list, y=[record[lam]['train_mse'] for lam in lambda_v_list], label='train')
        plt.scatter(x=lambda_v_list, y=[record[lam]['test_mse'] for lam in lambda_v_list], label='test')
        plt.xlabel('lambda value')
        plt.ylabel('MSE')
        plt.title('{} trian-test MSE plot'.format(group))
        plt.legend()
        plt.savefig(save_path + '{}_task1_plot.png'.format(group))
        # plt.show()
        task_1_min_lam_v = [v for v in record if record[v]['test_mse'] == np.min(
            [record[vi]['test_mse'] for vi in record])][0]
        task_1_min_test_mse = record[task_1_min_lam_v]['test_mse']
        record['best_lambda'] = task_1_min_lam_v
        record['best_mse'] = task_1_min_test_mse
        end_time = time.time() - start_time
        record['run_time'] = end_time
        total_record[group] = record
    np.save(save_path + 'task_1_record.npy', total_record, allow_pickle=True)


# Compare with true MSE

############################################
# Task 2


def stratifid_cross_validation(x, y, fold_n=10):
    # Do 10-fold stratified cross validation
    data = [x[i] + y[i] for i in range(len(x))]
    # Make permutation
    random_id = [id for id in range(len(data))]
    np.random.shuffle(random_id)
    new_f = {j: data[j] for j in random_id}
    cv_data = {ki: {} for ki in range(fold_n)}
    # In k-fold cross validation we generate k train/test splits
    for k in range(fold_n):
        test_start = int((k / fold_n) * len(data))
        test_end = int(((k + 1) / fold_n) * len(data))
        cv_data[k]['test_x'] = [new_f[id][:-1] for id in new_f if (id >= test_start) and (id < test_end)]
        cv_data[k]['test_y'] = [[new_f[id][-1]] for id in new_f if (id >= test_start) and (id < test_end)]
        cv_data[k]['train_x'] = [new_f[id][:-1] for id in new_f if (id < test_start)] + [new_f[id][:-1] for id in new_f
                                                                                         if (id >= test_end)]
        cv_data[k]['train_y'] = [[new_f[id][-1]] for id in new_f if (id < test_start)] + [[new_f[id][-1]] for id in
                                                                                          new_f if (id >= test_end)]
    return cv_data


def main_task_2(save_path='pp2/pp2data/pp2data/'):
    group_list = ['crime', 'wine', 'artsmall', 'artlarge']
    lambda_v_list = [v for v in range(1, 151)]
    min_record = {}
    task_1_record = np.load(save_path + 'task_1_record.npy', allow_pickle=True).item()
    for group in group_list:
        start_time = time.time()
        phi = open_csv_file('train-{}.csv'.format(group))
        t = open_csv_file('trainR-{}.csv'.format(group))
        cv_data = stratifid_cross_validation(x=phi, y=t, fold_n=10)
        lav_record = {}
        for lam_v in lambda_v_list:
            cv_record = {}
            for fold_i in range(10):
                tmp_phi = cv_data[fold_i]['train_x']
                tmp_t = cv_data[fold_i]['train_y']
                w = linear_regression_predict(tmp_phi, tmp_t, lam_v)
                mse_test = mse_main(cv_data[fold_i]['test_x'], cv_data[fold_i]['test_y'], w)
                cv_record[fold_i] = {'w': w, 'test_mse': mse_test}
            cv_record['mean'] = np.mean([cv_record[i]['test_mse'] for i in range(10)])
            lav_record[lam_v] = cv_record
        min_mse, respond_lam = float('Inf'), -1
        for lam_v in lav_record:
            if lav_record[lam_v]['mean'] < min_mse:
                min_mse = lav_record[lam_v]['mean']
                respond_lam = lam_v
        # Re-train on entire train set
        phi = open_csv_file('train-{}.csv'.format(group))
        t = open_csv_file('trainR-{}.csv'.format(group))
        test_data = open_csv_file('test-{}.csv'.format(group))
        test_data_R = open_csv_file('testR-{}.csv'.format(group))
        w = linear_regression_predict(phi, t, respond_lam)
        mse_train = mse_main(phi, t, w)
        mse_test = mse_main(test_data, test_data_R, w)
        # Compare with Task 1
        task_1_record_group = task_1_record[group]
        end_time = time.time() - start_time
        print('\n###################################')
        print('{}:'.format(group))
        print('Task 1: lambda={}, test MSE={}'.format(task_1_record_group['best_lambda'], task_1_record_group['best_mse']))
        print('Task 2: lambda={}, test MSE={}'.format(respond_lam, mse_test))
        min_record[group] = {'task1': {'lambda': task_1_min_lam_v, 'test_mse': task_1_min_test_mse},
                             'task2': {'lambda': respond_lam, 'test_mse': mse_test, 'run_time':end_time}}
    np.save(save_path + 'task_2_record.npy', min_record, allow_pickle=True)


##############################################################
# Task 3
import time


def main_task_3(save_path):
    group_list = ['crime', 'wine', 'artsmall', 'artlarge']
    task_record = {}
    for group in group_list:
        start_time = time.time()
        phi = open_csv_file('train-{}.csv'.format(group))
        t = open_csv_file('trainR-{}.csv'.format(group))
        alpha = 5
        beta = 5
        alpha_new = alpha - 2
        beta_new = beta - 2
        w_dim = len(phi[0])
        sample_n = len(phi)
        # phi = np.matrix(phi)
        phitphi = np.dot(np.transpose(phi), phi)
        phitphi_ = phitphi.copy()
        lam_v_init = np.linalg.eigvals(phitphi_)
        while np.abs(alpha - alpha_new) > 0.0001:
            alpha, beta = float(alpha_new), float(beta_new)
            lam_v = lam_v_init * beta
            phitphi_ = phitphi.copy()
            for i in range(len(phitphi_)):
                for j in range(len(phitphi_[i])):
                    phitphi_[i][j] *= beta

            S_n_inv = np.mat([[int(i == j) * alpha for j in range(w_dim)] for i in range(w_dim)]) + np.mat(phitphi_)
            S_n = np.linalg.inv(S_n_inv)
            m_n = beta * np.dot(S_n, np.dot(np.transpose(phi), t))
            gamma = np.sum([lam_v[k] / (lam_v[k] + alpha) for k in range(len(lam_v))])
            alpha_new = gamma / np.dot(np.transpose(m_n), m_n)
            mse_part = np.mat(t) - np.mat(np.dot(phi, [[float(i)] for i in m_n]))
            mse_part = np.sum([m ** 2 for m in mse_part])
            beta_new = 1 / ((1 / (sample_n - gamma)) * mse_part)
            print(alpha-alpha_new)
        alpha, beta = float(alpha_new), float(beta_new)
        lam_v = alpha / beta
        # phitphi_ = phitphi.copy()
        # for j in range(len(phitphi_[i])):
        #     phitphi_[i][j] *= beta
        # S_n_inv = np.mat([[int(i == j) * alpha for j in range(w_dim)] for i in range(w_dim)]) + np.mat(phitphi_)
        # S_n = np.linalg.inv(S_n_inv)
        # m_n_w = beta * np.dot(S_n, np.dot(np.transpose(phi), t))
        test_data = open_csv_file('test-{}.csv'.format(group))
        test_data_R = open_csv_file('testR-{}.csv'.format(group))
        mse_test = mse_main(test_data, test_data_R, m_n)
        end_time = time.time() - start_time
        print('\n################################')
        print('{}: MSE={}, lambda={}'.format(group, mse_test, lam_v))
        task_record[group] = {'test_mse': mse_test, 'lambda': lam_v, 'alpha': alpha, 'beta': beta, 'run_time': end_time}
    np.save(save_path + 'task_3_record.npy', task_record, allow_pickle=True)


#######################################################
# Task 4
def main_task_4(save_path):
    task1 = np.load(save_path + 'task_1_record.npy', allow_pickle=True).item()
    task2 = np.load(save_path + 'task_2_record.npy', allow_pickle=True).item()
    task3 = np.load(save_path + 'task_3_record.npy', allow_pickle=True).item()
    group_list = ['crime', 'wine', 'artsmall', 'artlarge']
    for group in group_list:
        print('\n###########################')
        print(group)
        print('task 1: best-lambda={},  test MSE={}, run-time={}'.format(task1[group]['best_lambda'], task1[group]['best_mse'], task1[group]['run_time']))
        print('task 2: best-lambda={},  test MSE={}, run-time={}'.format(task2[group]['task2']['lambda'], task2[group]['task2']['test_mse'], task2[group]['task2']['run_time']))
        try:
            print('task 3: best-lambda={},  test MSE={}, run-time={}'.format(task3[group]['lambda'], float(task3[group]['test_mse']), task3[group]['run_time']))
        except:
            print('task 3: best-lambda={},  test MSE={}, run-time={}'.format(task3[group]['lambda'], task3[group]['test_mse'], task3[group]['run_time']))


######################################################
######################################################
if __name__ == '__main__':
    save_path = 'pp2data/'
    # save_path = 'pp2/pp2data/pp2data/'
    print('##################################')
    print('Task 1: ')
    main_task_1(save_path)
    print('##################################')
    print('Task 2: ')
    main_task_2(save_path)
    print('##################################')
    print('Task 3: ')
    main_task_3(save_path)
    print('##################################')
    print('Task 4: ')
    main_task_4(save_path)
