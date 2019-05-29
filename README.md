# Colorization using Optimization(最优化上色)

## 介绍

============

彩色化灰度图是参考论文 [Levin, Lischinski & Weiss, 2004](http://www.cs.huji.ac.il/~yweiss/Colorization/colorization-siggraph04.pdf)在原网站给出了matlab版的代码，在目录matlab/origin中给出。同时在github上也找到了塔尔图大学的一个实现，在matlab/University of Tartu中。其中C++代码为本人的实现，其中加入了一些自己的改进。(未开始)

## 用法

============

1) 准备一张原始的灰度图，图像需要以RGB(3通道)格式保存。
2) 在原始图像上涂上任何颜色。可以使用任何喜欢的程序(例如:Photoshop,绘图)来生成涂鸦。确保不使用压缩，并且得到的涂鸦图片的RGB值与原始图像不同的只是彩色像素。(没太理解这部分)
3) 程序读入两张图片，生成采用最优化上色得到的彩色图片。

## 改进

============

## 参考资料

============

- [Levin, Lischinski & Weiss, paper, 2004](http://www.cs.huji.ac.il/~yweiss/Colorization/colorization-siggraph04.pdf)
- [Levin, Lischinski & Weiss, web](http://www.cs.huji.ac.il/~yweiss/Colorization/index.html)
- [Tartu University/github](https://github.com/geap/Colorization)

## ORIGINAL ARTICLE READ ME

============

This package contains an implementation of the image colorization approach described in the paper:
A. Levin D. Lischinski and Y. Weiss Colorization using Optimization.
ACM Transactions on Graphics, Aug 2004. 
 

Usage of this code is free for research purposes only. 
Please refer to the above publication if you use the program.

Copyrights: The Hebrew University of Jerusalem, 2004.
All rights reserved.

Written by Anat Levin.
Please address comments/suggestions/bugs to: alevin@cs.huji.ac.il
