#-------------------------------------------------------
#
# Module to support challenge.py
# Hayden Eastwood - 30-10-2018
# Last updated: 30-10-2018
# Version: 1.0
#
#
# -------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()

def batch_classify(X_train, Y_train, X_test, Y_test, dict_classifiers):  
    """ 
    Perform a batch classification of data:
    Input:
            X_train - training points
            Y_train - targets for training
            X_test - training for test points
            Y_test - target for training points
            dict_classifiers - the dictionary of classifiers to use
    """
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items()):
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff} 
        print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models



def display_dict_models(dict_models, sort_by='test_score'):
    """
    Display data from ML models 
    Input:
        dict_models - dictionary of models to display
    """
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    df_ = pd.DataFrame(data=numpy.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    print(df_.sort_values(by=sort_by, ascending=False))


def perf_measure(y_actual, y_hat, type="recall"):
    """
    Performance measure of y_actual against y_hat to determine TP, TN, FP, FN
    Input: 
        y_actual - test targets
        y_hat - predicted targets
        type = recall/specificity/full
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
			TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    if type == 'recall':
        output = TP/(TP+float(FN))
    elif type == 'specificity':
        output = FP/(FP+float(TN))
    elif type == 'full':
        output = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}    
    return(output) 


def plot_bar(stats):
    """ 
    Plot bar chart of key: value pairs in dictionary
    Input:
        stats: key: value pairs
    """
    y_pos = np.arange(len(stats))
    plt.bar(y_pos, stats.values(), align='center', alpha=0.5)
    plt.xticks(y_pos, stats.keys())
    plt.ylabel('Recall')
    plt.ylabel('Classifier')
    plt.title('Classifier fitness by recall')
    plt.show()
    

def plot_2d_space(X, y, label='Classes'):
    """
    Make scatter plot of data points X with class labels, y
    Input:
        X - input data points
        y - class labels 
    """
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X[y==l, 0],X[y==l, 1], c=c, label=l, marker=m)
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

