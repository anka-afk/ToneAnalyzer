# 功能介绍

本项目用于处理音频文件,操作步骤如下:

1. 取整段语音的平均能量
2. 取能量大于平均能量 8%的语音段并且保证该语音段持续时间在 0.05s-0.8s ,大于 0.8s 的段直接舍去
3. 对上一步每个片段取 30%-80%的时间段
4. 求上一步中取的区间中的 F0 均值【A1】、F1 均值、F2 均值、平均能量
5. 然后求各区段声调
6. 求步骤 3 中取的区间中的前 20%时间段的 F0 均值【A2】
7. 求步骤 3 中取的区间中的后 20%时间段的 F0 均值【A3】
   |A2-A1|<=A1*30% ，则为一声
   A2-A3>=A1*30%,则为四声
   不满足步骤 8、9，则为二三声
   输出: 表格文档,列为每个区间段的 F0 均值、F1 均值、F2 均值、声调、平均能量

# 项目依赖

参考
