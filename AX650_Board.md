http://10.126.35.128:25000/ 这是一个设备管理页面，里面有多种端侧设备。登陆密码可能是以下情况:
- 123456
- 123
- 0 

以下是另外的三块AX650N板子，登陆密码是 123456
Host AX650N-BJ0
    HostName 10.168.232.35
    User root

Host AX650N-BJ1
    HostName 10.168.232.117
    User root

Host AX650N-BJ2
    HostName 10.168.232.116
    User root

如果板端空间不够用，可以通过以下mout命令
```
mount -o nolock -t nfs 10.122.89.52:/ifs/data/tmp/yongqiang/nfs/ yongqiang
```
将板端的 /root/yongqiang 目录 挂载到 10.122.86.184的 /data/tmp/yongqiang/nfs/ 目录下，
然后登录 lihongjie@10.122.86.184，将文件放到 /data/tmp/yongqiang/nfs/lhj/{你新建的目录} ，在板端 /root/yongqiang/lhj/{你新建的目录} 就可以访问
