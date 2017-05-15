---
title: 生成博客过程中的小问题
categories: hexo_blog
tag: 
  - bolg
  - github
  - markdown
---

# 发布博客

hexo clean#清理

hexo g#generate,生成

hexo s#本地起个服务，预览

hexo deploy#发布

deploy的时候如果出现报错信息：ERROR Deployer not found: git

#解决办法
npm install hexo-deployer-git --save

<!--more-->

# 建立github免密

ssh-keygen -t rsa -b 4096 -C "mjp1124@126.com"

Enter a file in which to save the key (/home/you/.ssh/id_rsa): [Press enter]

Enter passphrase (empty for no passphrase): [Type a passphrase]

Enter same passphrase again: [Type passphrase again]

eval "$(ssh-agent -s)"

ssh-add ~/.ssh/id_rsa

sudo apt-get install xclip

xclip -sel clip < ~/.ssh/id_rsa.pub

去github,点击Settings

点击 SSH and GPG keys

点击 New SSH key or Add SSH key

在"Title"中可以对新的key添加一些描述，例如: "Personal MacBook Air"

在key的空格里面直接粘贴(因为前面的操作已经自动复制了)

点击Add SSH key

完成后需要重新登录

# notebook 转成markdowm

* jupyter nbconvert --to markdown test.ipynb --output /home/.../_posts/test2.md #单一文件转成md

* jupyter nbconvert --to markdown notebook的名字.ipynb --output-dir /home/.../_posts/ 

* 如果是批量转成md文件，用dir，但是不能改文件名字，单一文件可以不用加dir

* notebook 中的图片保存在notebook的名字_files文件中，需要手动把‘_files’删掉，需要保持md和文件夹名字一致

* md文件最后一行改成：{% asset_img 图片名称.png 想在博客中显示的图片名称 %}

* md文件中加入 <!--more-->,在博客中后面的内容显示为：阅读全文

* md的开头加入：
    > \---
    
    > title: 画图那些事儿
    
    > categories: pandas
    
    > tag: 
    
    >   \- plot
    
    >  \---
 

