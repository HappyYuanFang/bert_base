#coding:utf-8

from scipy.spatial import distance
from bert.extract_feature import BertVector
pooling_strategy = "REDUCE_MEAN"
#pooling_strategy = "NONE"
bc = BertVector(pooling_strategy=pooling_strategy, max_seq_len=80)
s1 = '人 同 去 福田 图书馆 啊 在 家 写 作业 巨 没 feel ， 我 的 作业'
s2 = "人同去福田图书馆啊在家写作业巨没feel，我的作业"
v = bc.encode([s1])
v1 = v["encodes"][0]
print(v1)
v = bc.encode([s2])
v2 = v["encodes"][0]
print(v2)
print(distance.cosine(v1,v2))
