# ONLY RUN FIRST TIME

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# IMPORTS

import random
import heapq
import numpy as np
import pandas as pd
import pickle as achar

import nltk
import gensim

import sklearn.model_selection as skms
import sklearn.svm as sksvm

# PARAMETERS

N_RESOURCES = 5
N_CERTS = 3
PERF_METRIC_THRESH = -0.5 # z-score

# UTILITY FUNCTIONS

def txt_to_vec(txt):
    # word selection utility function
    def nouns_and_adjs(txt):
        nouns = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(txt)) if pos[0] == 'N']
        adjs = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(txt)) if pos[0] == 'ADJ']
        return nouns + adjs

    # vector calculation
    feature = np.empty(100,)
    for noun_or_adj in nouns_and_adjs(txt):
        feature += MODEL[noun_or_adj]
    return feature

def n_closest(obj_lst, obj_target, n):
    # vector distance utility function
    def euc_dist(vec, target):
        return np.linalg.norm(vec - target)
    
    # distance calculation
    dist_lst, vt_dict, dv_dict = [], {}, {}
    target = txt_to_vec(obj_target.name)
    for obj in obj_lst:
        vec = txt_to_vec(obj.name)
        vt_dict[tuple(vec)] = obj
        dist = euc_dist(vec, target)
        dist_lst.append(dist)
        dv_dict[dist] = tuple(vec)
        
    # closest calculation (heapsort)
    closest = []
    heapq.heapify(dist_lst)
    size = len(dist_lst)
    while size > 0 and n > 0:
        next_dist = heapq.heappop(dist_lst)
        closest.append([vt_dict.get(dv_dict.get(next_dist, [0]*100), None), next_dist])
        n, size = n - 1, size - 1
    for _ in range(n):
        closest.append([None, 999])
    return closest

def cat_to_vec(categories_str):
    # implementation is weird because trying to optimize for time
    categories_lst = categories_str.split('#')
    vec, categories = [], ['ABCD of New IT', 'Behavioral and Business', 'Domain', 'Infosys Foundation Program', 'Infosys Internal', 'Marketing', 'Navigate the Change', 'New IT Foundations', 'Onboarding', 'Pentagon', 'Process', 'Projects', 'Project Practices', 'Sales', 'Technology']
    for category in categories:
        try:
            i = categories_lst.index(category)
        except:
            i = -1
        if i >= 0:
            vec.append(1)
            categories_lst.pop(i)
        else:
            vec.append(0)
    return vec

def dump_achar(obj, filename):
    fileobj = open(filename, 'wb')
    achar.dump(obj, fileobj)
    fileobj.close()

def load_achar(filename):
    fileobj = open(filename, 'rb')
    print(fileobj)
    obj = achar.load(fileobj, encoding='latin1')
    fileobj.close()
    return obj

# UTILITY CLASSES

class User:
    def __init__(self, utadf):
        # splitting into X dataframe and y dataframe
        test = 1 if utadf.size > 1 else 0
        Xdf, ydf = skms.train_test_split(utadf, test_size=test)
        self.res_y = Resource(ydf.iloc[0])
        
        # calculating n closest resources/certifications, using both distance between WordVecs and Course Name
        cour_lst, oth_lst, res_lst = [], [], []
        for X_i in range(Xdf.shape[0]):
            res_X = Resource(Xdf.iloc[X_i])
            if Xdf.iloc[X_i, 2] == self.res_y.name:
                cour_lst.append(res_X)
            else:
                oth_lst.append(res_X)
        if N_RESOURCES >= len(cour_lst):
            for res in cour_lst:
                res_lst.append([res, 0])
            res_lst.extend(n_closest(oth_lst, self.res_y, N_RESOURCES - len(cour_lst)))
        elif N_RESOURCES < len(cour_lst):
            res_lst.extend(n_closest(cour_lst, self.res_y, N_RESOURCES))
            
        # making X and y
        self.X, self.y = [], 1 if self.res_y.time_spent + self.res_y.score >= PERF_METRIC_THRESH else 0
        for res in res_lst:
            rX = res[0].rX() if res[0] is not None else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.X.extend(rX + [res[1]])
        self.X.extend(self.res_y.uX())

class Resource:
    def __init__(self, row):
        self.uuid = row[0]
        self.time_spent = row[1]
        self.course = row[2]
        self.name = row[3]
        self.cat = row[4]
        self.type = row[5]
        self.score = row[6]
    def rX(self):
        return [self.time_spent, self.score, ord(self.type)] + cat_to_vec(self.cat)
    def uX(self):
        return [ord(self.type)] + cat_to_vec(self.cat)
    def y(self):
        return self.time_spent

# DATA EXTRACTION

# read in excel files
tdf = pd.read_excel('timespent.xlsx')
adf = pd.read_excel('assessments.xlsx')
# drop all attempts but the latest
tdf.drop_duplicates(['uuid', 'course_name', 'resource_name'], inplace=True)
adf.drop_duplicates(['uuid', 'course_name'], inplace=True)
# DATA CLEANING

# clean merged time_spent and assessements datasets, split into train and test
tadf = tdf.merge(adf, how='left', on=['uuid', 'course_name'])
tadf = tadf.drop(['resource_date', 'assessment_date'], axis=1)
tadf.dropna(thresh=7, inplace=True)
tadf_train, tadf_test = skms.train_test_split(tadf, test_size=0.2)
# DATA ORDERING

# load word embedding model
MODEL = load_achar('fasttext_wv_lite')

# compile user list and dictionary
user_dict_train, user_lst_train = {}, []
user_dict_test, user_lst_test = {}, []

# for train set
for uuid, utadf_train in tadf_train.groupby('uuid'):
    if utadf_train.shape[0] > 1:
        user_train = User(utadf_train)
        user_dict_train[uuid] = user_train
        user_lst_train.append(user_train)

# for test set
for uuid, utadf_test in tadf_test.groupby('uuid'):
    if utadf_test.shape[0] > 1:
        user_test = User(utadf_test)
        user_dict_test[uuid] = user_test
        user_lst_test.append(user_test)

dump_achar(user_lst_train, 'user_lst_train')
dump_achar(user_lst_test, 'user_lst_test')

# DATA COMPILING

# for train set
X_train, y_train = [], []
for user_train in user_lst_train:
    X_train.append(user_train.X)
    y_train.append(user_train.y)

# for test set
X_test, y_test = [], []
for user_test in user_lst_test:
    X_test.append(user_test.X)
    y_test.append(user_test.y)

dump_achar(X_train, 'X_train')
dump_achar(y_train, 'y_train')
dump_achar(X_test, 'X_test')
dump_achar(y_test, 'y_test')

# MODEL TRAINING
svc = sksvm.SVC()
svc.fit(X_train, y_train)

# MODEL EVALUATION
print(svc.score(X_test, y_test))
dump_achar(svc, 'svc')

# NEXT STEPS (COURSE-RESOURCE DICTIONARY)

crdf = tdf.drop(['uuid', 'time_spent', 'resource_date', 'resource_category', 'resource_type'], axis=1)
crdict = {}
for course_name, cdf in crdf.groupby('course_name'):
    crdict[course_name] = cdf['resource_name'].unique()