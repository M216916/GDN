# preprocess data
import numpy as np
import re


def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:                                 # feature : M-6, M-1, M-2, S-2, P-10, ...
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())    # train : 1565 × 27 list ／ test : 2049 × 27 list
        else:                                                   # ×
            print(feature, 'not exist in data')                 # ×
    # append labels as last
    sample_n = len(res[0])                                      # train : 1565 ／ test : 2049

    if type(labels) == int:                                     # train のとき
        res.append([labels]*sample_n)                           # res に(0, 0, ..., 0) を追加 → 1565 × 28 list
    elif len(labels) == sample_n:                               # test のとき
        res.append(labels)                                      # res に(0, 1 の混合 )を追加 → 2049 × 28 list

    return res

def build_loc_net(struc, all_features, feature_map=[]):          # struc : 27組のfeature ／ all_features, feature_map : 27のfeature

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():                   # node_name : あるfeature ／ node_list : node_name以外のfeature
        
        if node_name not in all_features:                        # ×
            continue                                             # ×
        if node_name not in index_feature_map:                   # ×
            index_feature_map.append(node_name)                  # ×
        
        p_index = index_feature_map.index(node_name)             # p_index : 0, 1, 2, 3, ,..., 25, 26
        
        for child in node_list:                                  # node_list(node_name以外のfeature) の中のfeature 
            
            if child not in all_features:                        # ×
                continue                                         # ×
            if child not in index_feature_map:                   # ×
                print(f'error: {child} not in index_feature_map')# ×
                # index_feature_map.append(child)                # ×

            c_index = index_feature_map.index(child)             # child の index
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)                      # [1,2,3,...,25,26,0,2,3,...,25,26,0,1,3,...,25,26,...] 26child × 27
            edge_indexes[1].append(p_index)                      # [0,0,0,..., 0, 0,1,1,1,..., 1, 1,2,2,2,..., 2, 2,...] 26      × 27
        
    return edge_indexes                                          # (2, 702)
