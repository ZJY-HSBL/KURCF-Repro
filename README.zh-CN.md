# KURCF-Repro：KUR-CF 推荐算法独立复现

论文：Zhang 等，*Scientific Reports* **15**, 19600 (2025)。

https://doi.org/10.1038/s41598-025-96705-0

本项目并非作者的官方实现。论文中未公开的聚类初始化、预测公式、数据划分等细节采用公开透明的工程约定。目前**尚未证明论文报告的性能改善能够重复取得**。

## 实现内容

- Eq.3: 训练集用户电影评分矩阵 × 电影类型属性矩阵得到用户属性评分矩阵。
- Eq.4–6: 属性方差、方差权重、加权欧氏距离聚类。
- Eq.7: 聚类平均中心投影至最近观测样本。
- Eq.9–10: Pearson 与 Jaccard 组合相似度（alpha=0.3）。
- 按簇限制用户邻居，执行基于相似用户评分偏差的预测、推荐。
- 对比未聚类 UserCF 和聚类 Pearson 消融版本；提供 MAE、RMSE、Precision@K、Recall@K、轮廓系数分析。

## 安装与运行

~~~bash
python -m pip install -e ".[dev]"
pytest -q
kurcf download --dataset ml-100k
kurcf benchmark --dataset ml-100k --output results/100k.json
kurcf download --dataset ml-1m
kurcf benchmark --dataset ml-1m --output results/1m.json
kurcf sweep-k --dataset ml-100k
~~~

运行命令会从 GroupLens 官网下载 MovieLens 数据，不会将数据上传到 GitHub。默认对测试集的部分评分样本与用户计算指标；不可直接与论文的数字视为同一实验条件。

更多局限请参阅 [方法审计](docs/reproduction-notes.md)。
