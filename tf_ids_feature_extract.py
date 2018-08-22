from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

def txt_to_array(path):
    dir_f = os.listdir(path)

    big_dict = {}
    for dir_name in dir_f:
        if "." in dir_name:
            continue
        f_path = path + dir_name
        file = sorted(os.listdir(f_path))
        arr = []
        for f in file:
            f_name = path + dir_name + '/' + f
            res = ''
            with open(f_name, 'r') as ff:
                for i in ff:
                    res += i
            arr.append(res)
        big_dict[dir_name] = arr

    return big_dict

if __name__ == "__main__":
    path = "../gcca_data/three_view_data/"
    big_dict_str = txt_to_array(path)

    vec = TfidfVectorizer(min_df=2)
    big_dict = {}
    for k in big_dict_str.keys():
        arr = vec.fit_transform(big_dict_str[k])
        big_dict[k] = arr.toarray()
        print(arr.shape)
        # print(arr.toarray()[0])

    with open(path + "big_dict.pickle", 'wb') as f:
        pickle.dump(big_dict, f)


