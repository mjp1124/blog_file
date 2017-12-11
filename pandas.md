---
title: 利用pandas进行数据分析的常用小功能
categories: analysis
date: 2017-05-18
tags: 
  - pandas
  - data_analyse
---

本文主要介绍了，利用pandas进行数据处理时，经常会用到的一些小技巧

# 与时间相关


```python
import datetime as dt
from datetime import datetime 
import time 
```
<!--more-->
## 时间格式转成字符串
* 按照给定的格式（%Y%m%d）进行转化


```python
today = datetime.now()
today
```




    datetime.datetime(2017, 5, 19, 13, 48, 33, 86161)




```python
date_str  = datetime.strftime(today,'%Y%m%d')
date_str
```




    '20170519'



## 字符串格式转成时间
* 因为传入的字符串没有时间，所以自动转成了0点，传入是可以给定时间，注意后面的格式保持一致


```python
date = datetime.strptime('2017-05-15','%Y-%m-%d')
date
```




    datetime.datetime(2017, 5, 15, 0, 0)



## 浮点数转成时间

* 先用time.ctime转成时间格式后


```python
f_time=1.494484e+09
time.ctime(f_time)
```




    'Thu May 11 14:26:40 2017'



* 再用strptime转成想要的格式


```python
datetime.strptime(time.ctime(f_time),"%a %b %d %H:%M:%S %Y")
```




    datetime.datetime(2017, 5, 11, 14, 26, 40)



## 定义特定的时间


```python
#指定距离给定日期(today)n天的那个日期，例如n=7
today
```




    datetime.datetime(2017, 5, 19, 13, 48, 33, 86161)




```python
today-dt.timedelta(7)
```




    datetime.datetime(2017, 5, 12, 13, 48, 33, 86161)



## 其他

* 查询给定日期在今年是第几周，周几


```python
today.isocalendar()
```




    (2017, 20, 5)



# 切割

## 字符串的切割
* 用split返回的结果是个list


```python
test = 'ssssddddsfnnnnnn'
test.split('f')
```




    ['ssssdddds', 'nnnnnn']



* 以下定义了一个按照给定格式切割字符串的函数


```python
def parse_str(s,symbol):
    str1 = None
    str2 = None
    try:
        if pd.isnull(s):
            pass
        else:
            splited = s.split(symbol)
            if len(splited)==2:
                str1, str2 = splited
            else:
                str1, str2, *_ = splited #*_表示剩余部分

        return str1, str2
    except:
        return str1, str2
```


```python
test = 'ab|cd|ef'
parse_str(test,'|')
```




    ('ab', 'cd')




```python
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
```


```python
df = DataFrame({'blank':[test]*2})
df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>blank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ab|cd|ef</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ab|cd|ef</td>
    </tr>
  </tbody>
</table>
</div>

```python
df[['str1','str2']] = df['blank'].apply(lambda x:Series(parse_str(x,'|')))
df
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>blank</th>
      <th>str1</th>
      <th>str2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ab|cd|ef</td>
      <td>ab</td>
      <td>cd</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ab|cd|ef</td>
      <td>ab</td>
      <td>cd</td>
    </tr>
  </tbody>
</table>
</div>

## 数据的切割

* list中的数字切割


```python
test_data = [1,8,100]
pd.cut(test_data,[0,1,50,np.inf])
```




    [(0, 1], (1, 50], (50, inf]]
    Categories (3, object): [(0, 1] < (1, 50] < (50, inf]]



* 表格中的数字切割


```python
df = DataFrame({'number':test_data})
df
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>

```python
df['split_number'] = pd.cut(df['number'],[0,1,50,np.inf])
df
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>number</th>
      <th>split_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>(0, 1]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>(1, 50]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100</td>
      <td>(50, inf]</td>
    </tr>
  </tbody>
</table>
</div>

# 闭包的形式
* 对于内侧函数来说，如果某个变量一直是不变的，可以将此变量作为外层的参数，内层函数返回的是结果，外层函数返回的是内层函数，如下：
    * 此函数的用意：表中某列存了IP地址，根据IP地址返回城市
    * 外部有一个df中存放的是某段ip对应的城市信息


```python
def outer(df):
    def inner(ip):
        match=((df['start_ip_integer'].apply(lambda x:x<=ip)) & (df['end_ip_integer'].apply(lambda y:y>=ip)))
        return df[match]['city']
    return  inner

analyse_ip = outer(df)
```


```python
df['city']=df['ip'].apply(analyse_ip)
```

* region_df中存放的是经纬度对应的身份、城市，用一下函数返回


```python
def outer(region_df):
    def inner(df):
        try:
            longitude=df['longitude']
            latitude=df['latitude']
            matched_df=(region_df[(abs(region_df['longitude']-longitude)<0.01)&(abs(region_df['latitude']-latitude)<0.01)])

            if len(matched_df)>0:
                return matched_df[['province','city']].iloc[0,:]
            else:
                return Series([0,0],index = ['province','city'])
        except:
            print(matched_df)
    return  inner

analyse_orderid = outer(region_df)
```


```python
df[['orderid_pro','orderid_city']] = df[['longitude','latitude']].apply(analyse_orderid, axis=1)
```

# 添加文件


```python
import sys
sys.path.append('文件地址')
```

# 读写文件


```python
df = pd.read_excel('**.xls',)#读取excel文件
df = pd.read_csv('**.csv',)#读取csv文件
df = pd.read_csv('**.tsv', delimiter='\t')#读取tsv文件
df = pd.read_sql("SELECT * FROM table;",engine)#读取数据库中的数据
df.to_sql('table',engine,flavor='postgres',if_exists='replace',index=False)#将数据存入数据库
```

# 按轴转置
* unstack


```python
df= DataFrame({'kind1':['a','a','b','b'],'kind2':['k1','k2']*2,'num':[1,2,3,4]}).set_index(['kind1','kind2'])
df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>num</th>
    </tr>
    <tr>
      <th>kind1</th>
      <th>kind2</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>k1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>k2</th>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>k1</th>
      <td>3</td>
    </tr>
    <tr>
      <th>k2</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 直接转置的结果
df.T
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>kind1</th>
      <th colspan="2" halign="left">a</th>
      <th colspan="2" halign="left">b</th>
    </tr>
    <tr>
      <th>kind2</th>
      <th>k1</th>
      <th>k2</th>
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>num</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 用unstack
df.unstack('kind1')
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">num</th>
    </tr>
    <tr>
      <th>kind1</th>
      <th>a</th>
      <th>b</th>
    </tr>
    <tr>
      <th>kind2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>k1</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>k2</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

# 比较两个函数运行的时间，默认运行10000次


```python
from timeit import timeit
def func_compare(func_1,func_2,args=None,kwargs=None,n=10000):
    if args is None:
        args=[]
    if kwargs is None:
        kwargs={}
    print ("func_1 result:")
    print(func_1(*arges,**kwagrs))
    print()
    print ("func_2 result:")
    print(func_2(*arges,**kwagrs))
    print()
    
    t1=timeit(lambda:func_1(*arges,**kwagrs),number=n)
    t2=timeit(lambda:func_2(*arges,**kwagrs),number=n)
    print('time:{t1:0.3f}s vs time:{t2:0.3f}s'.format(t1=t1,t2=t2))
```

# while循环


```python
import pandas as pd
from pandas import DataFrame
from time import sleep
a=range(1,11)

step = 3
index_strart=0
index_end= index_strart+step

while True:
    index_true_end=min(index_end,len(a))
    s=list(a[index_strart:index_true_end])

    print(s)
    
    if (index_true_end<index_end) or (index_true_end==len(a)):
        break
        
    index_strart+=step
    index_end+=step 
```

    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]
    [10]


# 对列数据求和


```python
df = DataFrame({'kind':['kind1','kind2'],'count':[1,2],'amount':[20,30]})[['kind','count','amount']]
df
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>kind</th>
      <th>count</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kind1</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kind2</td>
      <td>2</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>

```python
df=df.set_index('kind').T
df
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>kind</th>
      <th>kind1</th>
      <th>kind2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>amount</th>
      <td>20</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>

```python
df['合计']=df.apply(sum,axis=1)
df
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>kind</th>
      <th>kind1</th>
      <th>kind2</th>
      <th>合计</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>amount</th>
      <td>20</td>
      <td>30</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>

```python
df=df.T.reset_index()
df
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>kind</th>
      <th>count</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kind1</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kind2</td>
      <td>2</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>合计</td>
      <td>6</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>

# 隐藏代码的语句


```python
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
```
