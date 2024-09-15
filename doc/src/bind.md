# Rust Bind

## Overview

总体来说，绑定rust的过程分成四步：

1. 将c++类封装成c类型的api
2. 使用xmake将c api和c++类编译成一个静态/动态库
3. rust方面使用build.rs来自动运行xmake，使用bindgen来自动生成绑定代码
4. 在rust代码中使用`include!(concat!(env!("OUT_DIR"), "/bindings.rs"));` 来引用c api，并用unsafe代码来调用这个api。

## 目前的问题

绑定过程中目前是把c++转为c，因此对模板的支持目前只能手动调整，可以考虑利用xmake里在编译时注入宏来完成模板参数从rust到c++的传递。也可以尝试使用bindgen对c++的支持，但官方文档对此的描述比较少。
