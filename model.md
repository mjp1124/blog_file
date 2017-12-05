---
title: 随机森林与GBDT简单对比及代码样例
categories: model
date: 2017-09-23
tags: 
  - model
  - Random Forest
  - GBDT
---

本文简单对比了随机森林与GBDT模型的优缺点，并列举了两种模型的样例的模型效果

# 随机森林(Random Forest)
* 随机森林指的是利用多棵树对样本进行训练并预测的一种分类器，随机森林的每一棵决策树之间是没有关联的

## 优缺点
### 优点

* 在数据集上表现良好
* 方差和偏差都比较低，泛化性能优越
* 在创建随机森林的时候，对generlization error使用的是无偏估计
* 它能够处理很高维度（feature很多）的数据，并且不用做特征选择
* 在训练完后，能够输出特征（feature）的重要性程度，非常实用    
* 高度并行化，易于分布式实现，训练速度快
* 在训练过程中，能够检测到feature间的互相影响
* 由于是树模型 ，不需要归一化即可之间使用，实现比较简单  
    
### 缺点
* 随机森林在某些噪音较大的分类或回归问题上会过拟合
* 分裂的时候，偏向于选择取值较多的特征
* 忽略属性之间的相关性
<!--more-->
## 代码样例


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pandas import DataFrame

model_data = pd.read_csv('model_data.csv')
```

### 查看数据情况（数据清洗过程已忽略）


```python
model_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Var1</th>
      <th>Var2</th>
      <th>Var3</th>
      <th>Var4</th>
      <th>Var5</th>
      <th>Var6</th>
      <th>Var7</th>
      <th>Var8</th>
      <th>Var9</th>
      <th>Var10</th>
      <th>...</th>
      <th>Var36</th>
      <th>Var37</th>
      <th>Var38</th>
      <th>Var39</th>
      <th>Var40</th>
      <th>Var41</th>
      <th>Var42</th>
      <th>Var43</th>
      <th>Var44</th>
      <th>tag_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>3</td>
      <td>703</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>3</td>
      <td>57</td>
      <td>2</td>
      <td>51</td>
      <td>91</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>46</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>-1</td>
      <td>3</td>
      <td>815</td>
      <td>2</td>
      <td>51</td>
      <td>-1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>5</td>
      <td>3</td>
      <td>354</td>
      <td>2</td>
      <td>51</td>
      <td>90</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>3</td>
      <td>391</td>
      <td>1</td>
      <td>61</td>
      <td>90</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>37</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
model_data.describe()
```
{% asset_img 描述.png 描述性统计 %}

### 变量相关性检验


```python
key = 'tag_y'
coef_df = []
for col in model_data.columns:
    corrcoef = np.corrcoef(model_data[col],model_data[key])[0,1]
    coef_df.append({'变量':col,'相关系数':corrcoef,'相关系数绝对值':abs(corrcoef)})
    
DataFrame(coef_df).sort_values('相关系数绝对值',ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>变量</th>
      <th>相关系数</th>
      <th>相关系数绝对值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44</th>
      <td>tag_y</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Var24</td>
      <td>-0.387025</td>
      <td>0.387025</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Var17</td>
      <td>0.210136</td>
      <td>0.210136</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Var28</td>
      <td>-0.157304</td>
      <td>0.157304</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Var23</td>
      <td>-0.150241</td>
      <td>0.150241</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Var16</td>
      <td>-0.095761</td>
      <td>0.095761</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Var21</td>
      <td>0.083199</td>
      <td>0.083199</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Var3</td>
      <td>-0.082515</td>
      <td>0.082515</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Var18</td>
      <td>-0.081642</td>
      <td>0.081642</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Var26</td>
      <td>-0.080799</td>
      <td>0.080799</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Var25</td>
      <td>-0.080647</td>
      <td>0.080647</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Var2</td>
      <td>-0.068508</td>
      <td>0.068508</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Var13</td>
      <td>-0.066173</td>
      <td>0.066173</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Var31</td>
      <td>-0.063315</td>
      <td>0.063315</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Var1</td>
      <td>0.062513</td>
      <td>0.062513</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Var14</td>
      <td>-0.058684</td>
      <td>0.058684</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Var19</td>
      <td>0.055990</td>
      <td>0.055990</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Var8</td>
      <td>-0.055442</td>
      <td>0.055442</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Var41</td>
      <td>0.051801</td>
      <td>0.051801</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Var40</td>
      <td>0.050829</td>
      <td>0.050829</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Var32</td>
      <td>0.044908</td>
      <td>0.044908</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Var37</td>
      <td>-0.042666</td>
      <td>0.042666</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Var29</td>
      <td>0.042470</td>
      <td>0.042470</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Var43</td>
      <td>0.042357</td>
      <td>0.042357</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Var34</td>
      <td>-0.033433</td>
      <td>0.033433</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Var35</td>
      <td>-0.031154</td>
      <td>0.031154</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Var33</td>
      <td>0.029990</td>
      <td>0.029990</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Var12</td>
      <td>-0.025125</td>
      <td>0.025125</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Var15</td>
      <td>0.024673</td>
      <td>0.024673</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Var10</td>
      <td>0.023796</td>
      <td>0.023796</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Var4</td>
      <td>-0.017289</td>
      <td>0.017289</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Var6</td>
      <td>-0.016744</td>
      <td>0.016744</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Var38</td>
      <td>-0.016197</td>
      <td>0.016197</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Var11</td>
      <td>0.015352</td>
      <td>0.015352</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Var5</td>
      <td>0.014266</td>
      <td>0.014266</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Var7</td>
      <td>-0.013188</td>
      <td>0.013188</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Var20</td>
      <td>-0.013174</td>
      <td>0.013174</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Var42</td>
      <td>-0.012301</td>
      <td>0.012301</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Var22</td>
      <td>-0.010746</td>
      <td>0.010746</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Var9</td>
      <td>0.010663</td>
      <td>0.010663</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Var36</td>
      <td>0.009791</td>
      <td>0.009791</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Var44</td>
      <td>-0.006633</td>
      <td>0.006633</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Var27</td>
      <td>-0.004636</td>
      <td>0.004636</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Var30</td>
      <td>0.002331</td>
      <td>0.002331</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Var39</td>
      <td>-0.000359</td>
      <td>0.000359</td>
    </tr>
  </tbody>
</table>
</div>



### 随机森林建模


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from numpy import array, linspace, infty

Y = array(model_data[key])
X = model_data.drop(key,axis=1)

## 数据切割成训练集和测试集（此处比例选择的是8:2）

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

## 模型的各项参数
model_config = {
    'model': RandomForestClassifier,
    #     'args':
    'kargs':{
        'n_estimators': 1000,
        'class_weight': 'balanced',
        'max_features': 'auto',
        'max_depth': 5,
        'min_samples_leaf': 100,
        'random_state':33,
        'bootstrap': True,
        'oob_score': True
    }
}

model = model_config['model']
other_args = model_config['kargs']

#告诉模型参数
clf = model(**other_args)
clf

## 训练模型，喂数据
clf = clf.fit(X_train, y_train)

clf.oob_score_ #验证集上的准确率（非测试集、非训练集）
```




    0.67633816908454225




```python
## 变量重要性输出
importance = pd.DataFrame({'dummy_variable':X_train.columns, 'importance':clf.feature_importances_})\
    .sort_values('importance', ascending=False)
importance['importance'] = importance['importance'].apply(lambda x: round(x, 3))

importance
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dummy_variable</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Var24</td>
      <td>0.435</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Var17</td>
      <td>0.127</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Var16</td>
      <td>0.090</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Var28</td>
      <td>0.076</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Var23</td>
      <td>0.053</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Var14</td>
      <td>0.024</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Var18</td>
      <td>0.021</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Var26</td>
      <td>0.020</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Var21</td>
      <td>0.019</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Var6</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Var27</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Var43</td>
      <td>0.012</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Var31</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Var3</td>
      <td>0.009</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Var42</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Var9</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Var1</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Var8</td>
      <td>0.006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Var2</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Var19</td>
      <td>0.004</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Var29</td>
      <td>0.004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Var4</td>
      <td>0.004</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Var13</td>
      <td>0.004</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Var12</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Var40</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Var25</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Var7</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Var15</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Var11</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Var22</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Var41</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Var37</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Var34</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Var10</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Var30</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Var44</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Var32</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Var38</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Var35</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Var33</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Var36</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Var39</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Var5</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Var20</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
## 测试数据的y预测值
y_pred_prob = clf.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve, auc, roc_auc_score,precision_recall_curve, f1_score

fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)# fpr:false positive rate;tpr:true positive rate
```

### AUC值和ROC曲线


```python
roc_auc = auc(fpr, tpr)
roc_auc
```




    0.79153905777374467




```python
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
from matplotlib.pyplot import plot, figure, rc
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "Microsoft YaHei"#"Droid Sans Fallback"#
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 20
```


```python
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()
plt.savefig('随机森林ROC曲线.png')
```

{% asset_img model_15_0.png 随机森林ROC曲线 %} 


### K-S值和K-S曲线


```python
ks = tpr-fpr

max(ks)
```




    0.47769207501512401




```python
plt.figure()
lw = 2
plt.plot(threshold, ks, color='darkorange',
         lw=lw, label='KS curve (max = %0.2f)' % max(ks))
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Threshold')
plt.ylabel('K-S value')
plt.title('K-S曲线')
plt.legend(loc="lower right")
plt.show()
plt.savefig('随机森林KS曲线.png')
```
{% asset_img model_18_0.png 随机森林KS曲线 %} 


### PRC曲线与f1_score曲线


```python
precision, recall, threshold_pr = precision_recall_curve(y_test, y_pred_prob)
f1_score_ = 2*recall*precision/(precision + recall)
```


```python
plt.figure()
lw = 2
plt.plot(recall,precision , color='darkorange',
         lw=lw, label='PRC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall Rate')
plt.ylabel('Precision Rate')
plt.title('PRC曲线')
plt.legend(loc="lower right")
plt.show()
plt.savefig('随机森林PRC曲线.png')
```

{% asset_img model_21_0.png 随机森林PRC曲线 %} 



```python
plt.figure()
lw = 2
plt.plot(threshold_pr, f1_score_[:-1] ,color='darkorange',
         lw=lw, label='f1_score curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('threshold')
plt.ylabel('f1_score')
plt.title('f1_score曲线')
plt.legend(loc="middle right")
plt.show()
plt.savefig('随机森林f1_score曲线.png')
```
{% asset_img model_22_0.png 随机森林f1_score曲线 %} 



```python
max(f1_score_)
```




    0.63074901445466491



### 阈值与留存


```python
from jfdata.utilities.model_evaluation import evaluate_binary_classifer
from jfdata.utilities.visualization import line_plot
```


```python
score_df = pd.DataFrame({'y':pd.Series(y_test).map({1:True, 0:False}), 'prob':y_pred_prob})

score_df[(True, True)] = 0
score_df[(True, False)] = 1
score_df[(False, False)] = 1
score_df[(False, True)] = 0

s = evaluate_binary_classifer(score_df, 20)

anatations =  ['{0:0.2f}%'.format(100*rate) for rate in s['target_rate']]
plot_x = s.threshold
plot_y = s.score
x_label = '风险度阈值'
y_label = '留存数量'
title = '不同的风险阈值对留存的正常用户和bad占比的影响'

f1 = line_plot(plot_x, plot_y, title, x_label, y_label, anatations)
plt.savefig('随机森林阈值.png')
```

{% asset_img model_26_0.png 随机森林阈值 %} 


# 梯度提升树GBDT (Gradient Boosting Decision Tree)

## 简介/备注
* GBDT的树都是回归树，而不是分类树，GBDT 是多棵树的输出预测值的累加
* 在Gradient Boost中，每个新的模型的建立是为了使得之前模型的残差往梯度方向减少
* GDBT 由损失函数和正则化函数两部分构成:
    * 损失函数尽可能的小，这样使得目标函数能够尽可能的符合样本
    * 正则化函数防止模型过拟合
    * 寻求损失函数和正则化函数的平衡点
* Feature 分裂原则：
    * 遍历所有Feature，找到每个Feature的增益最大的分裂点，并计算出每个Feature分裂点的增益
    * 取所有Feature分裂点的增益最大的Feature作为最先分裂点
    * 使用贪婪法，重复上面的过程，建立一棵完整的决策树    
* 每次分裂的目的是为了获得更多的信息增益，如果分裂后信息增益为负数，则停止分裂

## 优缺点
### 优点
* 预测精度高
* 能处理非线性数据、多特征类型
* 适合低维稠密数据
* 模型可解释行好?
* 可以灵活处理各种类型的数据，包括连续值和离散值，不需要做特征的归一化
* 在相对少的调参时间情况下，预测的准备率也可以比较高
* 使用一些健壮的损失函数，对异常值的鲁棒性非常强
### 缺点
* 弱学习器之间存在依赖关系，难以并行训练数据,是个串行的过程
* 计算复杂度大
* 不使用高维稀疏特征

gbdt使用什么损失函数？比如 Huber损失函数和Quantile损失函数；均方误差和LogLoss等

## 代码样例


```python
model_config = {
    'model': RandomForestClassifier,
    #     'args':
    'kargs':{
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_features': 5,
        'max_depth': 4,
        'min_samples_leaf': 100,
        'random_state':33,
    }
}

model = model_config['model']
other_args = model_config['kargs']
```

### GBDT建模


```python
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(**other_args)
gbdt.fit(X_train,y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=4,
                  max_features=5, max_leaf_nodes=None,
                  min_impurity_split=1e-07, min_samples_leaf=100,
                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                  n_estimators=100, presort='auto', random_state=33,
                  subsample=1.0, verbose=0, warm_start=False)



### 算法评估指标


```python
pred_y= gbdt.predict(X_test)#默认阈值为0.5

pd.crosstab(y_test,pred_y)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>row_0</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>629</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>147</td>
      <td>157</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pandas import DataFrame

print(classification_report(y_test, pred_y, digits=4))
```

                 precision    recall  f1-score   support
    
              0     0.8106    0.9037    0.8546       696
              1     0.7009    0.5164    0.5947       304
    
    avg / total     0.7772    0.7860    0.7756      1000
    


### ROC曲线


```python
y_pred_prob = gbdt.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)# fpr:false positive rate;tpr:true positive rate

roc_auc = auc(fpr, tpr)
roc_auc
```




    0.81887571839080453




```python
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()
plt.savefig('GBDT_ROC曲线.png')
```
{% asset_img model_38_0.png GBDT_ROC曲线 %} 


### K-S值和K-S曲线


```python
ks = tpr-fpr

max(ks)
```




    0.50457501512401692




```python
plt.figure()
lw = 2
plt.plot(threshold, ks, color='darkorange',
         lw=lw, label='KS curve (max = %0.2f)' % max(ks))
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Threshold')
plt.ylabel('K-S value')
plt.title('K-S曲线')
plt.legend(loc="lower right")
plt.show()
plt.savefig('GBDT_KS曲线.png')
```
{% asset_img model_41_0.png GBDT_KS曲线 %} 


### PRC曲线与f1_score曲线


```python
precision, recall, threshold_pr = precision_recall_curve(y_test, y_pred_prob)
f1_score_ = 2*recall*precision/(precision + recall)
```


```python
plt.figure()
lw = 2
plt.plot(recall,precision , color='darkorange',
         lw=lw, label='PRC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall Rate')
plt.ylabel('Precision Rate')
plt.title('PRC曲线')
plt.legend(loc="lower right")
plt.show
plt.savefig('GBDT_PRC曲线.png')
```

{% asset_img model_44_0.png GBDT_PRC曲线 %} 



```python
plt.figure()
lw = 2
plt.plot(threshold_pr, f1_score_[:-1] ,color='darkorange',
         lw=lw, label='f1_score curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('threshold')
plt.ylabel('f1_score')
plt.title('f1_score曲线')
plt.legend(loc="middle right")
plt.show()
plt.savefig('GBDT_f1_score曲线.png')
```

{% asset_img model_45_0.png GBDT_f1_score曲线 %}


### 阈值选取问题


```python
score_df = pd.DataFrame({'y':pd.Series(y_test).map({1:True, 0:False}), 'prob':y_pred_prob})

score_df[(True, True)] = 0
score_df[(True, False)] = 1
score_df[(False, False)] = 1
score_df[(False, True)] = 0

s = evaluate_binary_classifer(score_df, 20)

anatations =  ['{0:0.2f}%'.format(100*rate) for rate in s['target_rate']]
plot_x = s.threshold
plot_y = s.score
f1 = line_plot(plot_x, plot_y, title, x_label, y_label, anatations)
plt.savefig('GBDT阈值.png')
```

{% asset_img model_47_0.png GBDT阈值 %}


### 自动调参


```python
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

gbdt = GradientBoostingClassifier()
cross_validation = StratifiedKFold(pd.Series(y_train),n_folds = 10)
parameter_grid = {'max_depth':[2,3,4,5],
                  'max_features':[1,3,5,7,9],
                  'n_estimators':[100,300,500,1000]}
grid_search = GridSearchCV(gbdt,param_grid = parameter_grid,cv =cross_validation,scoring = 'accuracy')

grid_search.fit(X_train,pd.Series(y_train))
```




    GridSearchCV(cv=sklearn.cross_validation.StratifiedKFold(labels=[0 1 ..., 1 0], n_folds=10, shuffle=False, random_state=None),
           error_score='raise',
           estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_split=1e-07, min_samples_leaf=1,
                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                  n_estimators=100, presort='auto', random_state=None,
                  subsample=1.0, verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'n_estimators': [100, 300, 500, 1000], 'max_depth': [2, 3, 4, 5], 'max_features': [1, 3, 5, 7, 9]},
           pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)




```python
#输出最高得分
grid_search.best_score_
```

    0.7763881940970485

```python
#输出最佳参数
grid_search.best_params_
```

    {'max_depth': 5, 'max_features': 9, 'n_estimators': 100}


