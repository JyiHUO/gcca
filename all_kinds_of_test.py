from origin_gcca import *
from spare_gcca import *
from cca import *
from deep_cca import *
from wgcca import *
from dgcca_format import *
import numpy as np


def t_std_gene_data(clf_, n = 10):
    # read some data
    data = data_generate()
    name = ['Srbct', 'Leukemia', 'Lymphoma', 'Prostate', 'Brain', 'Colon']


    for i in range(6):
        train_corr = []
        test_corr = []
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
            v1_test, v2_test = clf.transform(data.test_data)
            train_corr.append(np.mean(clf.cal_correlation(clf.list_projection)))
            test_corr.append(np.mean(clf.cal_correlation([v1_test, v2_test])))
            train_acc.append(clf.cal_acc(clf.list_projection))
            test_acc.append(clf.cal_acc([v1_test, v2_test]))
            spare.append(clf.cal_spare()[0])

        print("std of train correlation is: ", np.std(train_corr), ' mean is ', np.mean(train_corr))
        print("std of test correlation is: ", np.std(test_corr),' mean is ', np.mean(test_corr))
        print("std of train acc is: ", np.std(train_acc), ' mean is ',np.mean(train_acc))
        print("std of test acc is: ", np.std(test_acc),' mean is ', np.mean(test_acc))
        print("std of spare is: ", np.std(spare), np.mean(spare))

        # print("|  %.3f(%.3f) |  %.3f(%.3f) | %.3f(%.3f) | %.3f(%.3f) | %.3f(%.3f) |"%(np.mean(test_corr),
        #                                                                               np.std(test_corr),
        #                                                                               np.mean(spare),
        #                                                                               np.std(spare),
        #                                                                               np.mean(train_acc),
        #                                                                               np.std(train_acc),
        #                                                                               np.mean(test_acc),
        #                                                                               np.std(test_acc),
        #                                                                               np.mean(train_corr),
        #                                                                               np.std(train_corr)))

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
        v1_test, v2_test = clf.transform(data.test_data)
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

            # train  model
            clf = clf_(ds=data, m_rank=2)
            clf.solve()

            # calculate all kind of metric
            v1_test, v2_test = clf.transform(data.test_data)
            print("total correlation in training data is: ", np.mean(clf.cal_correlation(clf.list_projection)))
            print("total correlation in testing data is: ", np.mean(clf.cal_correlation([v1_test, v2_test])))
            print("training data ACC is: ", clf.cal_acc(clf.list_projection))
            print("testing data ACC is: ", clf.cal_acc([v1_test, v2_test]))
            print("each view's spare of U is ", clf.cal_spare())
            print("total sqare is: ", clf.cal_spare()[0])

            print()
            print()

def t_three_view(clf_):
    data = data_generate()

    data.generate_three_view_tfidf_dataset()

    clf = clf_(ds=data, m_rank=20)
    clf.solve()

    # calculate all kind of metric
    print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
    print("each view's spare of U is ", clf.cal_spare())
    print("total sqare is: ", np.mean(clf.cal_spare()))

    print()
    print()

def t_synthetic_data(clf_):
    data = data_generate()

    data.generate_synthetic_dataset()

    clf = clf_(ds=data, m_rank=2)
    clf.solve()

    # calculate all kind of metric
    print("reconstruction error of G in training is: ", clf.cal_G_error(data.train_data, test=False))
    print("reconstruction error of G in testing is: ", clf.cal_G_error(data.test_data, test=True))
    print("each view's spare of U is ", clf.cal_spare())
    print("total sqare is: ", np.mean(clf.cal_spare()))

    print()
    print()
    return clf.cal_G_error(data.test_data, test=True)



if __name__ == "__main__":
    # change the model below you like

    # preserve all model in one list
    print("Which model do you want to choose?\n0.gcca,\n1.spare_gcca,\n2.cca,\n3.deepcca,\n4.WeightedGCCA,\n5.dgcca_")
    choose = int(input(">>>"))


    clf_list = [gcca, spare_gcca, cca, deepcca, WeightedGCCA, dgcca_]

    # choose which model you want to use
    clf_ = clf_list[choose]

    if choose in {0, 1, 2, 3}:

        print("Which result do you want to test?\n0.test in gene data,\n1.test in gene data for std,\n2.test in gene data whether normalize or not,\n4. nothing")
        choose1 = int(input(">>>"))

        if choose1 == 0:
            print ("################### start testing result #####################")
            print()
            t_result_gene_data(clf_)
            print("################### finish testing result #####################")
            print()

        if choose1 == 1:
            print("################### start testing std #####################")
            print()
            t_std_gene_data(clf_)
            print("################### finish testing std #####################")
            print()

        if choose1 == 2:
            print("################### start testing normalize or not #####################")
            print()
            t_if_normalize_or_not(clf_)
            print("################### finish testing normalize or not #####################")
            print()

    if choose in {0, 1, 4, 5}:
        print("Which result do you want to test?\n0.three view for language data,\n1.synthetic data\n2.nothing")
        choose1 = int(input(">>>"))

        if choose1 == 0:
            print("################### start testing three views of language data #####################")
            print()
            t_three_view(clf_)
            print("################### finish testing three views of language data #####################")
            print()

        if choose1 == 1:
            print("###################start testing synthetic data #####################")
            print()
            t_synthetic_data(clf_)
            print("###################finish testing synthetic data #####################")
            print()