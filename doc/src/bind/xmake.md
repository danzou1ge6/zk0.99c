# xmake编译

xmake编译总体和原来没有什么区别，但需要注意现在要编译成lib。

## 动态lib

只需要加入`  set_kind("shared")` 即可。

## 静态lib

首先需要加入`  set_kind("static") `，但是由于默认情况下xmake并不会自动将nvcc生成的一些代码链接进静态库，而由于我们后续的rust编译中完全不涉及到nvcc，因此就会产生报错，因此需要`  add_values("cuda.build.devlink", true)` 来手动开启devlink。
