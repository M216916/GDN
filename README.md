# モデル
### GDN_3
    
# データ
### yfinance_5
* list.txt  train.csv  test.csv  →  回帰
* true.csv  x_non                →  分類
        
# 実行手順
* 埋込ベクトル x     : torch.Size([716,  64])
* 非時系列属性 x_non : torch.Size([716,  40])
* 統合(axis=1) x     : torch.Size([716, 104])
* Light GBM で分類

# 結果
### 非時系列なし
        val_acc=0.4266  test_acc=0.5033
### 非時系列あり
        val_acc=0.4777  test_acc=0.4099
