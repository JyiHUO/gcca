from origin_gcca import *
import numpy as np

def t_times( n = 10):
    # read some data
    data = data_generate()


    for i in range(6):
        corr = []
        train_acc = []
        test_acc = []
        spare = []


        print()
        print("finish reading data: ",i)
        print()

        for j in range(n):
            data.generate_genes_data(num=i, random_state=np.random.randint(1, 100, 1)[0])
            # train gcca model
            clf = gcca(ds=data, m_rank=2)
            clf.solve()

            # calculate all kind of metric
            corr.append(np.mean(clf.cal_correlation()))
            train_acc.append(clf.cal_acc(clf.list_projection))
            test_acc.append(clf.cal_acc(clf.transform(data.test_data[0].T, data.test_data[1].T)))
            spare.append(clf.cal_spare()[0])

        print ("std of correlation is: ", np.std(corr))
        print ("std of train acc is: ", np.std(train_acc))
        print("std of test acc is: ", np.std(test_acc))
        print("std of spare is: ", np.std(spare))

        print()
        print()



t_times(n=30)