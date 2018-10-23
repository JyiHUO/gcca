import pandas as pd
import pickle
from collections import OrderedDict

def read_ff():
    sentences = pd.read_csv('../gcca_data/multi_language/sentences_detailed.csv',
     sep='\t',
     header= None,
        names =  ['id', 'lang', 'sentence', 'user', 'date1', 'date2'])
    links = pd.read_csv('../gcca_data/multi_language/links.csv',
        sep='\t', names = ['source', 'target'])
    return sentences, links

def select_id(lang_list, num_entry, sentences, links):
    f_id = sentences[sentences.lang == lang_list[0]].id
    big_arr = []
    c = 0
    # f_id = f_id.sample(f_id.shape[0])

    for i in f_id:
        target = links[links.source == i].target
        lang_check = {lang_list[i]: False for i in range(len(lang_list))}
        lang_sens = OrderedDict({lang_list[i]: "" for i in range(len(lang_list))})

        lang_sens[lang_list[0]] = sentences[sentences.id == i].sentence.values[0]
        lang_check[lang_list[0]] = True
        for t in target:
            temp = sentences[sentences.id == t]
            if temp.shape[0] != 1:
                break
            if (temp.lang.values[0] in set(lang_list[1:])) and (lang_check[temp.lang.values[0]] == False):
                lang_sens[temp.lang.values[0]] = temp.sentence.values[0]
                lang_check[temp.lang.values[0]] = True

        # check
        if sum(lang_check.values()) != len(lang_list):
            continue
        big_arr.append(list(lang_sens.values()))
        c+=1
        print(c)
        if c == num_entry:
            break


    df = pd.DataFrame(columns=lang_list, data=big_arr)

    return df

sens, link = read_ff()
lang_list = ["eng", "tur", "epo", "cmn"]
num_entry = 1000
df = select_id(lang_list=lang_list, num_entry=num_entry, sentences=sens, links=link)

name = "_".join(lang_list)
df.to_csv(index=False,path_or_buf="../gcca_data/multi_language/"+name+".csv")

