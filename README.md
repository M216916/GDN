# モデル
### GDN_2
    
# データ
### yfinance_5
* list.txt  train.csv  test.csv  →  回帰
* true.csv  x_non                →  分類
        
# 実行手順
* 埋込ベクトル x     : torch.Size([716,  64])
* 非時系列属性 x_non : torch.Size([716,  40])
* 統合(axis=1) x     : torch.Size([716, 104])
* Neural Network で分類（716 → 5 → 3）

# 結果
### 埋込vec ＋ 非時系列 → 104カラム
        val_acc=0.4777  test_acc=0.4099
### 埋込vec → 64カラム
        val_acc=0.4266  test_acc=0.5033
### 非時系列 → 40カラム
        val_acc=0.5511  test_acc=0.5300
