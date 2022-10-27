# モデル
### GDN_3
    
# データ
### yfinance_5
* list.txt ／ train.csv ／ test.csv  →  回帰
* true.csv ／ x_non                →  分類
        
# 実行手順
* 埋込ベクトル x     : torch.Size([716,  64])
* 非時系列属性 x_non : torch.Size([716,  40])
* 統合(axis=1) x     : torch.Size([716, 104])
* Light GBM で分類

# 結果
### 埋込vec ＋ 非時系列
    [17 29  3]
    [22 49  1]
    [ 7 13  3]
    accuracy      :0.479167
    【0】precision:0.369565  recall:0.346939  F1:0.357895
    【1】precision:0.538462  recall:0.680556  F1:0.601227
    【2】precision:0.428571  recall:0.130435  F1:0.200000
### 埋込vec
    [19 30  0]
    [17 55  0]
    [ 6 17  0]
    accuracy      :0.513889
    【0】precision:0.452381  recall:0.387755  F1:0.417582
    【1】precision:0.539216  recall:0.763889  F1:0.632184
    【2】precision: nan  recall:0.000000  F1: nan
### 非時系列
    [10 39  0]
    [15 57  0]
    [ 5 18  0]]
    accuracy      :0.465278
    【0】precision:0.333333  recall:0.204082  F1:0.253165
    【1】precision:0.500000  recall:0.791667  F1:0.612903
    【2】precision: nan  recall:0.000000  F1: nan
