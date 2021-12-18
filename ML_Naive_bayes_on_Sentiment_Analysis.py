'''
ABuild Naive Bayes algorithm with maximum likelihood and MAP solutions and evaluate it using cross validation on the task of sentiment analysis.
'''
import math
import os
import csv
from math import sqrt, log, inf
import random
import plotly
import plotly.graph_objs as go
from plotly.offline import plot


########################################
def mean_data(num):
    """
    :param num: list
    :return: float
    """
    if not len(num):
        print('input length is zero.')
        return
    return sum(num) / len(num)


def stdev_data(num):
    x_bar = mean_data(num)
    est_variance = sum([(x - x_bar) ** 2 for x in num]) / float(len(num) - 1)
    return sqrt(est_variance)


def open_txt_file(file_name='amazon_cells_labelled.txt', base_dir='program_project/pp1data/pp1data/'):
    """
    :param file_name: str
    :param base_dir: str, where the file is saved
    :return: list
    """
    f1 = open(base_dir + file_name, 'r', encoding="utf-8")
    f_sample = []
    for line in f1:
        f_sample.append([line.split('\t')[0], int(line.split('\t')[1][0])])
    f_sample = [s for s in f_sample if s != '']
    return f_sample


def symbol_process(sentence):
    sentence = sentence.replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('\x96', '').replace('"', '')
    return sentence


def get_features(f, show=True):
    """
    :param f: list, [[x0, y0], [x1, y1],...], output of open_txt_file()
    :return:
    """
    seperate_words = []
    feature_list = {}
    for sample_i in f:
        words = symbol_process(sample_i[0]).split(' ')
        if not len(words):
            continue
        words = [w.lower() for w in words]
        seperate_words.append([words, sample_i[1]])

    label_count = {}
    for sample in seperate_words:
        if sample[1] not in label_count:
            label_count[sample[1]] = {'n': 1}
        else:
            label_count[sample[1]]['n'] += 1
    label_list = list(label_count.keys())

    if not min([label_count[k]['n'] for k in label_list]):
        print('\nLack of enough labelled data')
        return
    for label in label_count:
        label_count[label]['ratio'] = label_count[label]['n'] / sum([label_count[x]['n'] for x in label_count])
        label_count[label]['token_n'] = 0
    for sample_i in seperate_words:
        for wi in sample_i[0]:
            if wi not in feature_list:
                feature_list[wi] = {'N': 1}
                for label in label_list:
                    feature_list[wi]['{}_n'.format(label)] = int(sample_i[1] == label)
            else:
                feature_list[wi]['N'] += 1
                feature_list[wi]['{}_n'.format(sample_i[1])] += 1
            label_count[sample_i[1]]['token_n'] += 1

    if '' in feature_list:
        del feature_list['']
    if show:
        print('number of features: {}'.format(len(feature_list)))
    for feature in feature_list:
        for label in label_list:
            feature_list[feature]['MaxL_{}_p'.format(label)] = feature_list[feature]['{}_n'.format(label)] / label_count[label]['token_n']
    return feature_list, seperate_words, label_count


def predict_model(feature_list, test_set, label_count, model='MaxL', show=True):
    assert (model == 'MaxL') or (model.startswith('MAP_m'))
    seperate_words_test = []
    for sample_i in test_set:
        words = symbol_process(sample_i[0]).split(' ')
        if not len(words):
            continue
        words = [w.lower() for w in words]
        seperate_words_test.append([words, sample_i[1]])
    for i in range(len(seperate_words_test)):
        label_score = {label_i: 0 for label_i in label_count}
        for label in label_count:
            for feature in seperate_words_test[i][0]:
                if feature not in feature_list:
                    continue
                if feature_list[feature]['{}_{}_p'.format(model, label)] == 0:
                    label_score[label] += -inf
                else:
                    label_score[label] += log(feature_list[feature]['{}_{}_p'.format(model, label)])
            label_score[label] = label_count[label]['ratio'] * label_score[label]
        predict_label = max(label_score, key=label_score.get)
        seperate_words_test[i] += [predict_label, int(predict_label == seperate_words_test[i][1])]
    accuracy_score = mean_data([c[3] for c in seperate_words_test])
    if show:
        print('{} accuracy={}'.format(model, accuracy_score))
    return seperate_words_test, accuracy_score


def dirichlet_prior_process(feature_list, label_count, dirichlet_m=1):
    if type(dirichlet_m) is str:
        dirichlet_m = float(dirichlet_m)
    # Add dirichlet prior, m=dirichlet_m
    for label in label_count:
        label_count[label]['new_token_n'] = label_count[label]['token_n'] + dirichlet_m * len(feature_list)
    for feature in feature_list:
        for label in label_count:
            feature_list[feature]['MAP_m{}_{}_p'.format(dirichlet_m, label)] = (feature_list[feature]['{}_n'.format(label)] + dirichlet_m) /  label_count[label]['new_token_n']
    return feature_list, label_count


def seperate_train_test_set(f, train_ratio=0.7):
    """
    :param f: list
    :param train_ratio: float, the ratio of train set in total data set
    :return:
    """
    train_set, test_set = [], []
    for s in range(len(f)):
        randnum = random.random()
        if randnum <= train_ratio:
            train_set.append(f[s])
        else:
            test_set.append(f[s])
    return train_set, test_set


def stratifid_cross_validation(f, fold_n=10):
    # Do 10-fold stratified cross valuidation
    new_f = {}
    # Seperating the initial samples according to theri labels
    for line in f:
        if line[1] not in new_f:
            new_f[line[1]] = [line]
        else:
            new_f[line[1]].append(line)
    # Make permutation
    for label in new_f:
        random.shuffle(new_f[label])
    cv_data = {ki: {'train':[], 'test':[]} for ki in range(fold_n)}
    # In k-fold cross validation we generate k train/test splits
    for k in range(fold_n):
        for label in new_f:
            test_start = int((k / fold_n) * len(new_f[label]))
            test_end = int(((k + 1) / fold_n) * len(new_f[label]))
            cv_data[k]['test'] += new_f[label][test_start:test_end]
            cv_data[k]['train'] += new_f[label][:test_start]
            cv_data[k]['train'] += new_f[label][test_end:]
    return cv_data


def main_3_1( file='amazon_cells_labelled.txt', base_dir='program_project/pp1data/pp1data/'):
    """
    3.1 Naive Bayes for Text Categorization
    :return:
    """
    print('==================================================')
    print('File: {}'.format(file))
    raw_data_set = open_txt_file(file_name=file, base_dir=base_dir)
    train_set, test_set = seperate_train_test_set(f=raw_data_set, train_ratio=0.7)
    ###################
    # Train Set part
    ###################
    feature_list, seperate_words, label_count = get_features(train_set)
    ###################
    # Test Set part
    ###################
    seperate_words_test, acc_score = predict_model(feature_list, test_set, label_count, model='MaxL')
    return raw_data_set, feature_list, label_count, seperate_words_test, acc_score,  train_set, test_set


def main_3_2(test_set, feature_list, label_count, dirichlet_m=1):
    # We use the output from main_3_1
    print('==================================================')
    ###################
    # Add dirichlet prior into train set features
    ###################
    feature_list, label_count = dirichlet_prior_process(feature_list, label_count, dirichlet_m=dirichlet_m)
    ###################
    # Test Set part
    ###################
    seperate_words_test, acc_score = predict_model(feature_list, test_set, label_count, model='MAP_m{}'.format(dirichlet_m))


def main_3_3(raw_data_set):
    # use output from main_3_1
    # out put cv_f is a dict with key=fold
    cv_f = stratifid_cross_validation(raw_data_set, fold_n=10)
    return cv_f


def main_3_4(cv_f, fold_n=10, model_list=None, size_j_list=None, show=True):
    # use output from main_3_3
    print('==================================================')
    if model_list is None:
        model_list = ['MAP_m1']
    if size_j_list is None:
        size_j_list = [j/10 for j in range(1, 11)]
    score_record = {k:{model:{j/10: 0 for j in range(1, 11)} for model in model_list} for k in range(fold_n)}
    for k in range(fold_n):  # in each fold
        if show:
            print('================================')
            print('for {}-th fold:'.format(k))
        data_k = cv_f[k]
        random.shuffle(data_k['train'])
        random.shuffle(data_k['test'])
        for size_j in  size_j_list:  # in different size
            if show:
                print('-------------------------------------------')
                print('train size = {} N'.format(size_j))
            total_n = len(data_k['train'])
            train_set_j = data_k['train'][:int(total_n * size_j)]
            test_set_j = data_k['test']
            feature_list, seperate_words, label_count = get_features(train_set_j, show=show)
            for model in model_list:
                # Add dirichlet prior
                dirichlet_mi = model.split('_m')[1]
                if '.' in dirichlet_mi:
                    dirichlet_mi = float(dirichlet_mi)
                else:
                    dirichlet_mi = int(dirichlet_mi)
                feature_list, label_count = dirichlet_prior_process(feature_list, label_count, dirichlet_m=dirichlet_mi)
                # predict on test set
                seperate_words_test_j, acc_score_j = predict_model(feature_list, test_set_j, label_count, model=model, show=show)
                score_record[k][model][size_j] += acc_score_j
    # Calculate average and stddev for each size
    summary_record = {model: {sizej: {'average': 0, 'stddev': 0} for sizej in size_j_list} for model in model_list}
    for model in model_list:
        for j_size in summary_record[model]:
            summary_record[model][j_size]['average'] = mean_data([score_record[k][model][j_size] for k in score_record])
            summary_record[model][j_size]['stddev'] = stdev_data([score_record[k][model][j_size] for k in score_record])
        print('Learning Curve for {}:'.format(model))
        for j_size in summary_record[model]:
            print('{} N: average={}, stddev={}'.format(j_size, round(summary_record[model][j_size]['average'], 3),
                                                       round(summary_record[model][j_size]['stddev'], 3)))
    return summary_record


def save_to_csv(file_name, write_list):
    with open(file_name, 'w') as myfile:
        mywr = csv.writer(myfile, lineterminator='\n')
        for item in write_list:
            mywr.writerow(item)
    myfile.close()


def main_4_experiments(base_dir='program_project/pp1data/pp1data/'):
    print('####################################')
    print('Part 1 of Experiments')
    for file in ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']:
        print('==================================================')
        print('File: {}'.format(file))
        raw_data_set = open_txt_file(file_name=file, base_dir=base_dir)
        cv_f = stratifid_cross_validation(raw_data_set, fold_n=10)
        learning_curve_dict = main_3_4(cv_f=cv_f, fold_n=10, model_list=['MAP_m0', 'MAP_m1'], show=False)
        ######################################
        # save as csv file
        write_list = [['model', 'train_size(N)', 'average_accuracy', 'accuracy_stddev']]
        for model in learning_curve_dict:
            for size_j in learning_curve_dict[model]:
                write_list.append([model, size_j, learning_curve_dict[model][size_j]['average'], learning_curve_dict[model][size_j]['stddev']])
        save_to_csv(file[:-4] + '_learning_curve_m1m2.csv', write_list)
        ######################################
        # plot
        data = []
        for model in learning_curve_dict:
            data.append(
                go.Scatter(
                    x=list(learning_curve_dict[model].keys()),
                    y=[learning_curve_dict[model][c]['average'] for c in learning_curve_dict[model]],
                    error_y=dict(type='data', array=[learning_curve_dict[model][c]['stddev'] for c in learning_curve_dict[model]], visible=True),
                    name='average accuracy (m={})'.format(model.split('_m')[1])
                )
            )
        layout = go.Layout(
            title='{}: learning curve'.format(file[:-4]),
            xaxis=dict(title='train size'),
            yaxis=dict(title='accuracy')
        )
        fig = go.Figure(data=data, layout=layout)
        plot_url = plot(fig, filename='learning_curve_{}_m1m2.html'.format(file[:-4]))
    #######################################
    print('####################################')
    print('Part 2 of Experiments')
    for file in ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']:
        print('==================================================')
        print('File: {}'.format(file))
        raw_data_set = open_txt_file(file_name=file, base_dir=base_dir)
        cv_f = stratifid_cross_validation(raw_data_set, fold_n=10)
        total_m_list = [g/10 for g in range(10)] + [d for d in range(1, 11)]
        model_list = ['MAP_m{}'.format(h) for h in total_m_list]
        learning_curve_dict = main_3_4(cv_f, fold_n=10, model_list=model_list, size_j_list=[1], show=False)
        # save to csv
        write_list = [['m', 'cv_average_accuracy', 'cv_accuracy_stddev']]
        for mi in total_m_list:
            model_i = 'MAP_m{}'.format(mi)
            write_list.append([mi, learning_curve_dict[model_i][1]['average'], learning_curve_dict[model_i][1]['stddev']])
        save_to_csv(file[:-4] + '_smoothing_parameter.csv', write_list)
        # plot
        data = [
            go.Scatter(
                x=total_m_list,
                y=[learning_curve_dict[model][1]['average'] for model in learning_curve_dict],
                error_y=dict(type='data',
                             color='gray',
                             array=[learning_curve_dict[model][1]['stddev'] for model in learning_curve_dict],
                             visible=True),
                name='cross validation accuracy'
            )
        ]
        layout = go.Layout(
            title='{}: cross validation accuracy and stddev'.format(file[:-4]),
            xaxis=dict(title='smoothing parameter'),
            yaxis=dict(title='accuracy')
        )
        fig = go.Figure(data=data, layout=layout)
        plot_url = plot(fig, filename='smoothing_parameter_{}.html'.format(file[:-4]))


#################################################
#################################################
#################################################
if __name__ == '__main__':
    main_4_experiments(base_dir='pp1data/')







