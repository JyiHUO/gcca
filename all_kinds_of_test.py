from origin_gcca import *
from spare_gcca import *
from cca import *
from deep_cca import *
import numpy as np


def t_std_gene_data(clf_, n = 10):
    # read some data
    data = data_generate()
    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']


    for i in range(6):
        corr = []
        train_acc = []
        test_acc = []
        spare = []


        print()
        print("finish reading data: ",name[i])
        print()

        for j in range(n):

            # split data using different random_state
            data.generate_genes_data(num=i, random_state=np.random.randint(1, 100, 1)[0])

            # train gcca model
            clf = clf_(ds=data, m_rank=2)
            clf.solve()

            # calculate all kind of metric
            corr.append(np.mean(clf.cal_correlation()))
            train_acc.append(clf.cal_acc(clf.list_projection))
            test_acc.append(clf.cal_acc(clf.transform(data.test_data[0].T, data.test_data[1].T)))
            spare.append(clf.cal_spare()[0])

        print("std of correlation is: ", np.std(corr))
        print(corr)
        print("std of train acc is: ", np.std(train_acc))
        print("std of test acc is: ", np.std(test_acc))
        print("std of spare is: ", np.std(spare))

        print()
        print()

def t_result_gene_data(clf_):
    # read some data
    data = data_generate()
    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']

    for i in range(6):
        data.generate_genes_data(num=i)

        print()
        print("finish reading data: ", name[i])
        print()

        # train gcca model
        clf = clf_(ds=data, m_rank=2)
        clf.solve()

        # calculate all kind of metric
        v1_test, v2_test = clf.transform(data.test_data[0], data.test_data[1])
        print("total correlation in training data is: ", np.mean(clf.cal_correlation(clf.list_projection)))
        print("total correlation in testing data is: ", np.mean(clf.cal_correlation([v1_test, v2_test])))
        print("training data ACC is: ", clf.cal_acc(clf.list_projection))
        print("testing data ACC is: ", clf.cal_acc([v1_test, v2_test]))
        print("each view's spare of U is ", clf.cal_spare())
        print("total sqare is: ", clf.cal_spare()[0])

        print()
        print()

def t_if_normalize_or_not(clf_):
    # read some data
    data = data_generate()
    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']

    for c in [True, False]:
        print("$$$$$$$$$$$normalize is: ", c)
        for i in range(6):
            data.generate_genes_data(num=i, normalize=c)

            print()
            print("finish reading data: ", name[i])
            print()

            # train gcca model
            clf = clf_(ds=data, m_rank=2)
            clf.solve()

            # calculate all kind of metric
            v1_test, v2_test = clf.transform(data.test_data[0], data.test_data[1])
            print("total correlation in training data is: ", np.mean(clf.cal_correlation(clf.list_projection)))
            print("total correlation in testing data is: ", np.mean(clf.cal_correlation([v1_test, v2_test])))
            print("training data ACC is: ", clf.cal_acc(clf.list_projection))
            print("testing data ACC is: ", clf.cal_acc([v1_test, v2_test]))
            print("each view's spare of U is ", clf.cal_spare())
            print("total sqare is: ", clf.cal_spare()[0])

            print()
            print()



if __name__ == "__main__":
    # change the model below you like

    # preserve all model in one list
    clf_list = [gcca, spare_gcca, cca, deepcca]

    # choose which model you want to use
    clf_ = clf_list[3]  # gcca

    print ("###################start testing result#####################")
    print()
    t_result_gene_data(clf_)
    print("###################finish testing result#####################")
    print()

    print("###################start testing std#####################")
    print()
    t_std_gene_data(clf_)
    print("###################finish testing std#####################")
    print()

    print("###################start testing normalize or not#####################")
    print()
    t_if_normalize_or_not(clf_)
    print("###################finish testing normalize or not#####################")
    print()