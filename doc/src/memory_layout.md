# Memory Layout

Halo2中使用的是64位整数作为一个WORD，而cuda中使用的是32位整数作为一个WORD，但由于两者都是little endian，而目标系统中的整数布局也是little endian的，因此在内存中的布局可以认为是完全相同的，因此域中的整数从Halo2到cuda并不需要做转换。

![image-20240915120825413](memory_layout.assets/image-20240915120825413.png)