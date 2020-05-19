---
layout:     post                    # 使用的布局（不需要改）
title:      springboot中踩的雷               # 标题 
subtitle:    #副标题
date:       2020-05-19              # 时间
author:     serenity                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
	- Java
	- Springboot
---

### 本帖子用于记录springboot使用中踩的雷

#### springboot接受string并自动转化枚举类型

background: controller层接收了一个前端传来的对象类型（user），对象类型里包含一一个枚举类（usertype），

```java
//controller层

@PostMapping("/register")
    public ResponseVO registerAccount(@RequestBody UserVO userVO) {
        return accountService.registerAccount(userVO);
    }

//usertype

package com.example.hotel.enums;
public enum UserType {
    Client("1"),
    HotelManager("2"),
    Manager("3");
    private String value;

    UserType(String value) {
        this.value = value;
    }
    @Override
    public String toString() {
        return value;
    }

}

//user相关part
public class UserVO {
    private UserType userType;
    public UserType getUserType() {
        return userType;
    }
    public void setUserType(UserType userType) {
        this.userType = userType;
    }
```

可以看到，这里将传入的string对应为enum的过程都是由springboot自行完成的。而springboot自带的converter埋下的一个坑就在这里：**它是通过下标，而非value，来进行转化的**。以上述情况为例，如果前端传来的usertype为"1"，实际创建的不是Client（value为1），而是HotelManager（下标为1）。因此，springboot也就无法应对更复杂的enum情况，如

```java
public enum CourseType {

    PICTURE(102, "图文"),
    AUDIO(103, "音频"),
    VIDEO(104, "视频");

    private final int index;
    private final String name;
}
```

如果要实现通过value转换，可参考下面两篇blog

https://www.cnblogs.com/coderxiaohei/p/12835852.html

https://blog.csdn.net/lanqibaoer/article/details/62215380

