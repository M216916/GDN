# モデル
### GDN_5 ※pushしてみる
    
# データ
### yfinance_5
* 2730社
* list.txt ／ train.csv ／ test.csv  →  回帰
* true.csv ／ x_non                →  分類
        
# 実行手順
* 埋込ベクトル x     : torch.Size([716,  64])
* 非時系列属性 x_non : torch.Size([716,  40])
* 統合(axis=1) x     : torch.Size([716, 104])
* Light GBM で分類
* slide:40 dim:32
* 部分時系列にaverage適用

# 結果

