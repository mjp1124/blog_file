---
title: 评价分类模型好坏的重要指标ROC,PRC,KS
categories: model
date: 2017-12-12
tags: 
  - model
  - Random Forest
  - GBDT
  - analysis
---

前段时间整理了一篇关于随机森林和GBDT模型对比的文章，里面在建模的过程中，用到了很多评价模型的指标，所以此次将这些指标单独拿出来进行一下概括，主要包括ROC曲线和AUC值，PRC曲线和f1值，KS曲线和KS值
在将上述内容时，首先明确以下指标：
* TP：正确肯定的数目
* FN：漏报，没有正确找到的匹配的数目
* FP：误报，给出的匹配是不正确的
* TN：正确拒绝的非匹配对数
<!--more-->
{% asset_img tf.png 图１ %}
* ROC曲线
    * ROC曲线的横坐标为false positive rate（FPR），纵坐标为true positive rate（TPR）
    * TPR和FPR的计算公式分别为：
    	* 真阳性率（真正类率） TPR = TP / (TP + FN)
    	* 伪阳性率（负正类率 ）FPR = FP / (FP + TN)
	* AUC值：ROC曲线下的面积，取值范围为0.5-1，AUC值越大，模型的分类效果越好
    * ROC曲线例如下
{% asset_img roc.png ROC曲线 %}
* PRC曲线
	* PRC曲线是以precision为横坐标，recall为纵坐标
    * PRC曲线例如下
{% asset_img PRC曲线.png  PRC曲线 %}
	* precision和recall的计算公式分别为：
      * 精度(precision)  P = TP / (TP + FP)
      * 召回率(recall)  R = TP / (TP + FN)
	* 精度和召回率是一对矛盾的变量，将二者结合起来，成为另一个变量F1值
	* F1 = 2PR / (P+R)  此时认为精准度和召回率的权重是一样的
	* 利用点调整模型比利用线调整模型方便的多，所以，选取f1的最大点作为阈值更便利一些，f1曲线图如下：
{% asset_img f1_score.png  f1_score %}

* ROC和PRC曲线对比
	* 当正负样本差距不大的情况下，ROC曲线和P-R的趋势是差不多的
	* 当负样本很多的时候，ROC曲线效果依然较好，但是PRC曲线效果一般

* KS曲线
	* KS曲线是正样本洛伦兹曲线和负样本洛伦兹曲线的差值曲线，KS曲线的最高点定义为KS值
	* KS是检验阳性与阴性分类区分能力的指标
    * KS曲线如下
{% asset_img KS曲线.png  KS曲线 %}

