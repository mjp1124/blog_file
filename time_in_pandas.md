---
title: 常用操作小笔记
categories: analysis
date: 2018-08-15
tags: 
  - git
  - screen
---
* 本文是将数据分析工作中一些常用的关于时间的技巧进行提炼总结，主要参考以下资料
    * 书籍：利用python进行数据分析
    * 网站：http://www.runoob.com/python/python-date-time.html

# 关于时间的一些定义

## 格式定义
* 与系统环境无关
    * %Y 四位年
    * %y 两位年
    * %m 两位月
    * %d 两位日
    * %H 时（24小时制）
    * %I 时（12小时制）
    * %M 两位分
    * %S 两位秒
    * %w 用整数表示的星期几（0-6）
    * %U 每年的第几周，从0开始（周日为第一天）
    * %W 每年的第几周，从0开始（周一为第一天）
    * %F %Y-%m-%d的简写
    * %D %m/%d/%y的简写
<!--more-->
* 与系统环境有关
    * %a 星期几的简写
    * %A 星期几的全称
    * %b 月份的简写
    * %B 月份的全称
    * %c 完整的日期和时间
    * %p 不同环境的am或pm
    * %x 适用于当前环境的日期格式
    * %X 适用于当前环境的时间格式

## 时间序列的基础频率
* D   ：   每日历日
* B   ：   每工作日
* H   ：   每小时
* T/min   ：   每分
* S   ：   每秒
* L/ms   ：   每毫秒
* U   ：   每微秒
* M   ：   每月最后一个日历日
* BM   ：   每月最后一个工作日
* MS   ：   每月第一个日历日
* BMS   ：   每月第一个工作日
* W-MON、W-TUE...   ：   从指定的星期几开始算起，每周
* WOM-1MON、WOM-2MON...   ：   产生每月第一、第二、第三、第四周的星期几
* Q-JAN、Q-FEB...   ：   对于以指定月份结束的年度，每季度最后一月的最后一个日历日
* BQ-JAN、BQ-FEB...   ：   对于以指定月份结束的年度，每季度最后一月的最后一个工作日
* QS-JAN、QS-FEB...   ：   对于以指定月份结束的年度，每季度最后一月的第一个日历日
* BQS-JAN、BQS-FEB...   ：   对于以指定月份结束的年度，每季度最后一月的第一个工作日
* A-JAN、A-FEB...   ：   每年指定月份的最后一个日历日
* BA-JAN、BA-FEB...   ：   每年指定月份的最后一个工作日
* AS-JAN、AS-FEB...   ：   每年指定月份的第一个日历日
* BAS-JAN、BAS-FEB...   ：   每年指定月份的第一个工作日

# pandas中关于时间的用法

* datetime模块中的数据类型：date,time,datetime,timedelat

## datetime.datetime
* datetime 以毫秒的形式，存储时间和日期
* 可以从datetime输出相应的年月日时间点等信息，分别为(year,month,day,hour,minute,second)


```python
from datetime import datetime
```


```python
now = datetime.now()
now
```




    datetime.datetime(2018, 8, 14, 17, 47, 57, 343367)




```python
now.year,now.month,now.day
```




    (2018, 8, 14)



## timedelta表示时间间隔
* 输出结果为日和秒，若秒=0，被忽略输出,可用days，seconds输出日，秒
* 日期与timedelta合用，可以输出指定时间间隔的时间


```python
delta = (datetime(2018,8,14,1,0,0)-datetime(2017,1,2,0,0,0))
delta
```




    datetime.timedelta(589, 3600)




```python
delta.days ,delta.seconds
```




    (589, 3600)




```python
delta = (datetime(2018,1,1)-datetime(2017,1,1))
delta
```




    datetime.timedelta(365)




```python
from datetime import timedelta
```


```python
datetime(2017,1,1) + timedelta(365)
```




    datetime.datetime(2018, 1, 1, 0, 0)



## datetime与字符串的相互转化
* str ，strftime
* strptime，parse ，to_datetime

### datetime to str


```python
stamp = datetime(2018,1,1)
```


```python
str(stamp)
```




    '2018-01-01 00:00:00'




```python
stamp.strftime('%Y-%m-%d')
```




    '2018-01-01'




```python
datetime.strftime(stamp,'%Y%m%d')
```




    '20180101'



### str to datetime
* strptime 可以把指定格式的str转化成日期进行输出
* parse 可以自动识别格式并进行输出（不支持中文格式），有些格式是日在前面的，可以用参数dayfirst控制
* pandas 中大部分是处理成组日期，用to_datetime


```python
stamp = '20180101'
```


```python
datetime.strptime(stamp,'%Y%m%d')
```




    datetime.datetime(2018, 1, 1, 0, 0)




```python
from dateutil.parser import parse
```


```python
parse('2018-01-01 12:00:05')
```




    datetime.datetime(2018, 1, 1, 12, 0, 5)




```python
parse('6/1/2017',dayfirst=True)
```




    datetime.datetime(2017, 1, 6, 0, 0)




```python
import pandas as pd
```


```python
test = ['20180101','20180102' , None]
```


```python
pd.to_datetime(test)
```




    DatetimeIndex(['2018-01-01', '2018-01-02', 'NaT'], dtype='datetime64[ns]', freq=None)



## date_range时间切片
* 主要的参数：
* pd.date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, **kwargs)
* start: 开始时间
* end: 结束时间
* periods: 周期,即输出个数
* freq: 时间间隔
* normalize: 是否规范化到午夜


```python
for date in pd.date_range('20180101',periods=5,freq='1d'):
    print(date)
```

    2018-01-01 00:00:00
    2018-01-02 00:00:00
    2018-01-03 00:00:00
    2018-01-04 00:00:00
    2018-01-05 00:00:00



```python
import numpy as np
```


```python
s = pd.Series(np.random.randn(10),pd.date_range('20180101',periods=10))
```


```python
s
```




    2018-01-01    0.458544
    2018-01-02    2.031096
    2018-01-03    0.608081
    2018-01-04   -0.030694
    2018-01-05    1.294916
    2018-01-06   -1.502905
    2018-01-07    1.263865
    2018-01-08    0.551192
    2018-01-09   -2.216100
    2018-01-10   -1.189207
    Freq: D, dtype: float64




```python
s['20180102':'20180105']
```




    2018-01-02    2.031096
    2018-01-03    0.608081
    2018-01-04   -0.030694
    2018-01-05    1.294916
    Freq: D, dtype: float64



## resample重采样

* series.resample(rule, how=None, axis=0, fill_method=None, closed=None, label=None, convention='start', kind=None, loffset=None, limit=None, base=0, on=None, level=None)

* 主要参数说明
    * freq： 表示重采样频率的字符串或DateOffset,例如'M'、'5min'或 Second(15)
    * how='mean' : 用于产生聚合值的函数名或数组函数，例如'mean'  'ohlc'.np.max等。默认为'mean'.
    * axis=0 : 其他常用的值有: 'first'、 'last'、 'median'、'ohlc'. 'max'、 'min'重采样的轴，默认为axis=0
    * fill_method=None : 升采样时如何插值，比如'fil'或'bill',默认不插值
    * closed='right' : 在降采样中，各时间段的哪一端是闭合(即包含)的，'right'或"left'。默认为right'
    * label='right' : 在降采样中，如何设置聚合值的标签，'right'或'left' (面元的右边界或左边界)。例如，9:30到9:35之间的这5分钟会被标记为9:30或9:35.默认为'right' 


```python
index = pd.date_range('2008-01-01', periods=10, freq='d')
series = pd.Series(range(10), index=index)
series
```




    2008-01-01    0
    2008-01-02    1
    2008-01-03    2
    2008-01-04    3
    2008-01-05    4
    2008-01-06    5
    2008-01-07    6
    2008-01-08    7
    2008-01-09    8
    2008-01-10    9
    Freq: D, dtype: int64



### 降低频率


```python
series.resample('3d').sum()
```




    2008-01-01     3
    2008-01-04    12
    2008-01-07    21
    2008-01-10     9
    Freq: 3D, dtype: int64




```python
series.resample('3d',label='right').sum()
```




    2008-01-04     3
    2008-01-07    12
    2008-01-10    21
    2008-01-13     9
    Freq: 3D, dtype: int64




```python
series.resample('3d',label='right',closed='left').sum()
```




    2008-01-04     3
    2008-01-07    12
    2008-01-10    21
    2008-01-13     9
    Freq: 3D, dtype: int64




```python
series.resample('3d',label='right',closed='right').sum()
```




    2008-01-01     0
    2008-01-04     6
    2008-01-07    15
    2008-01-10    24
    Freq: 3D, dtype: int64



### 按给定频率补充缺失


```python
series.resample('12H').asfreq()
```




    2008-01-01 00:00:00    0.0
    2008-01-01 12:00:00    NaN
    2008-01-02 00:00:00    1.0
    2008-01-02 12:00:00    NaN
    2008-01-03 00:00:00    2.0
    2008-01-03 12:00:00    NaN
    2008-01-04 00:00:00    3.0
    2008-01-04 12:00:00    NaN
    2008-01-05 00:00:00    4.0
    2008-01-05 12:00:00    NaN
    2008-01-06 00:00:00    5.0
    2008-01-06 12:00:00    NaN
    2008-01-07 00:00:00    6.0
    2008-01-07 12:00:00    NaN
    2008-01-08 00:00:00    7.0
    2008-01-08 12:00:00    NaN
    2008-01-09 00:00:00    8.0
    2008-01-09 12:00:00    NaN
    2008-01-10 00:00:00    9.0
    Freq: 12H, dtype: float64




```python
#用上一个值填充
series.resample('12H').pad()
```




    2008-01-01 00:00:00    0
    2008-01-01 12:00:00    0
    2008-01-02 00:00:00    1
    2008-01-02 12:00:00    1
    2008-01-03 00:00:00    2
    2008-01-03 12:00:00    2
    2008-01-04 00:00:00    3
    2008-01-04 12:00:00    3
    2008-01-05 00:00:00    4
    2008-01-05 12:00:00    4
    2008-01-06 00:00:00    5
    2008-01-06 12:00:00    5
    2008-01-07 00:00:00    6
    2008-01-07 12:00:00    6
    2008-01-08 00:00:00    7
    2008-01-08 12:00:00    7
    2008-01-09 00:00:00    8
    2008-01-09 12:00:00    8
    2008-01-10 00:00:00    9
    Freq: 12H, dtype: int64




```python
#用下一个值填充
series.resample('12H').bfill()
```




    2008-01-01 00:00:00    0
    2008-01-01 12:00:00    1
    2008-01-02 00:00:00    1
    2008-01-02 12:00:00    2
    2008-01-03 00:00:00    2
    2008-01-03 12:00:00    3
    2008-01-04 00:00:00    3
    2008-01-04 12:00:00    4
    2008-01-05 00:00:00    4
    2008-01-05 12:00:00    5
    2008-01-06 00:00:00    5
    2008-01-06 12:00:00    6
    2008-01-07 00:00:00    6
    2008-01-07 12:00:00    7
    2008-01-08 00:00:00    7
    2008-01-08 12:00:00    8
    2008-01-09 00:00:00    8
    2008-01-09 12:00:00    9
    2008-01-10 00:00:00    9
    Freq: 12H, dtype: int64




```python
series
```




    2008-01-01    0
    2008-01-02    1
    2008-01-03    2
    2008-01-04    3
    2008-01-05    4
    2008-01-06    5
    2008-01-07    6
    2008-01-08    7
    2008-01-09    8
    2008-01-10    9
    Freq: D, dtype: int64



### 截取


```python
series.truncate(after='2008-01-04')
```




    2008-01-01    0
    2008-01-02    1
    2008-01-03    2
    2008-01-04    3
    Freq: D, dtype: int64




```python
import time
```


```python
datetime.strptime(time.ctime(1234567890),'%a %b %d %H:%M:%S %Y')
```




    datetime.datetime(2009, 2, 14, 7, 31, 30)



## period


```python
p = pd.Period('2008',freq='A-JUN')
```


```python
p.asfreq('M',how='start')
```




    Period('2007-07', 'M')




```python
p.asfreq('M',how='end')
```




    Period('2008-06', 'M')




```python
P = pd.period_range('2001','2003',freq='A-NOV')
```


```python
P
```




    PeriodIndex(['2001', '2002', '2003'], dtype='period[A-NOV]', freq='A-NOV')




```python
P.asfreq('M',how='start')
```




    PeriodIndex(['2000-12', '2001-12', '2002-12'], dtype='period[M]', freq='M')




```python
P.asfreq('M',how='end')
```




    PeriodIndex(['2001-11', '2002-11', '2003-11'], dtype='period[M]', freq='M')




```python
len(P)
```




    3




```python
ts = pd.Series(np.random.randn(len(P)),index=P)
```


```python
ts
```




    2001   -0.450338
    2002   -0.340768
    2003    0.155203
    Freq: A-NOV, dtype: float64




```python
ts.asfreq('M',how='start')
```




    2000-12   -0.450338
    2001-12   -0.340768
    2002-12    0.155203
    Freq: M, dtype: float64




```python
ts.asfreq('M',how='end')
```




    2001-11   -0.450338
    2002-11   -0.340768
    2003-11    0.155203
    Freq: M, dtype: float64




```python
rng = pd.date_range('200101',periods=10,freq='M')
```


```python
rng
```




    DatetimeIndex(['2001-01-31', '2001-02-28', '2001-03-31', '2001-04-30',
                   '2001-05-31', '2001-06-30', '2001-07-31', '2001-08-31',
                   '2001-09-30', '2001-10-31'],
                  dtype='datetime64[ns]', freq='M')




```python
ts = pd.Series(range(10),index=rng)
```


```python
ts
```




    2001-01-31    0
    2001-02-28    1
    2001-03-31    2
    2001-04-30    3
    2001-05-31    4
    2001-06-30    5
    2001-07-31    6
    2001-08-31    7
    2001-09-30    8
    2001-10-31    9
    Freq: M, dtype: int64




```python
pts = ts.to_period()
```


```python
pts
```




    2001-01    0
    2001-02    1
    2001-03    2
    2001-04    3
    2001-05    4
    2001-06    5
    2001-07    6
    2001-08    7
    2001-09    8
    2001-10    9
    Freq: M, dtype: int64



## time 

### asctime
* 时间数组转成可读形式


```python
t1 = time.localtime()
```


```python
t1
```




    time.struct_time(tm_year=2018, tm_mon=8, tm_mday=16, tm_hour=14, tm_min=16, tm_sec=26, tm_wday=3, tm_yday=228, tm_isdst=0)




```python
time.asctime(t1)
```




    'Thu Aug 16 14:16:26 2018'




```python
t2 = (2018,5,27,1,5,27,6,147,-1)
```


```python
time.asctime(t2)
```




    'Sun May 27 01:05:27 2018'



### ctime
* 无参数是，返回当前时间点的字符串
* 传入参数时，把按秒计算的浮点数时间戳转化成asctime形式
* 可利用strptime转化成datetime格式


```python
time.ctime()
```




    'Thu Aug 16 14:35:01 2018'




```python
time.ctime(12345678)
```




    'Sun May 24 05:21:18 1970'




```python
datetime.strptime(time.ctime(12345678),'%a %b %d %H:%M:%S %Y')
```




    datetime.datetime(1970, 5, 24, 5, 21, 18)



### gmtime
* 用法与ctime类似，按秒计算的浮点数时间戳转化成struct_time


```python
time.gmtime()
```




    time.struct_time(tm_year=2018, tm_mon=8, tm_mday=16, tm_hour=6, tm_min=39, tm_sec=25, tm_wday=3, tm_yday=228, tm_isdst=0)




```python
time.gmtime(12345678)
```




    time.struct_time(tm_year=1970, tm_mon=5, tm_mday=23, tm_hour=21, tm_min=21, tm_sec=18, tm_wday=5, tm_yday=143, tm_isdst=0)



### loacltime
* 格式化时间戳为本地的时间


```python
localtime = time.localtime()
```


```python
a.tm_year
```




    2018




```python
localtime
```




    time.struct_time(tm_year=2018, tm_mon=8, tm_mday=16, tm_hour=14, tm_min=51, tm_sec=0, tm_wday=3, tm_yday=228, tm_isdst=0)




```python
localtime.tm_mday
```




    16


