---
title: 常用操作小笔记
categories: git
date: 2018-05-01
tags: 
  - git
  - screen
---
工作过程中，常用到的一些小操作，做个备份便于查询

### 服务与文件
* 连接服务器： [ssh user_name]@[server_ip]
* 在连接的服务器启动jupyter后再本地浏览器打开： jupyter notebook --ip=[server_ip] nohop ?
* 传送文件： scp -pr [本地文件夹/文件] [目标存放地址]
* 重启服务器: sudo reboot 
* 虚拟环境
    * 安装: pip3 install virtualenv
    * 新建: virtualenv -p [虚拟环境存放的目标位置]
    * 启动: source [目标位置/bin/activate]　
    * 自己编写的函数包，可以放在虚拟环境下的lib/python环境中的site-packages中
    * 指定调用的包的路径：site-packages mylib.pth中编写
* bashrc文件定义短句样例: alias jter='jupyter notebook'
<!--more-->
### git
* 比较文件区别： git diff [文件１]　[文件2] #git diff master比较的是历史区和工作区的差异
* 查看版本号： git log --graph
* 回退版本： git reset --hard [版本号]　#版本号前几个字段即可
* 查看每次操作的号段： git reflog
* 删除文件： git rm
* 克隆git上的文件: git clone 
* 分支(branch)相关
    * 查看分支: git branch
    * 新建分支: git checkout -b [branchname]
    * 切换分支: git checkout [branchname]
    * 合并分支: git merge [branchname]
    * 删除分支: git branch　-d [branchname]
* 拉取git上的文件: git pull origin [branchname]
* 文件修改后传到git上
    * git status
    * git add --all
    * git commit -m [备注信息]
    * git push origin [branchname]
* 连接远程仓库: git remote add origin [仓库的地址]

### screen
* 创建并进入: screen
* 创建一个新的运行shell的窗口并切换到该窗口: Ctrl-a c
* 暂时离开,后台执行: Ctrl-a d #会输出一个screen地址
* 进入制定地址的window: screen -r [screen地址]
* 关闭：　Ctrl-a k
* 显示所有键绑定信息: Ctrl-a ?
* 切换到下一个window: Ctrl-a n
* 切换到前一个window:　Ctrl-a p
