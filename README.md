# spark-recommender
使用Spark实现一些常用的推荐算法。

### 样本数据data.db
#### TB_ORDER,用户对商品的购买记录30w

UID | PID | ORDER_COUNT|
| :------: | :------: | :------: |
用户id | 商品id | 购买数量|


- ALS矩阵分解
```
com.lxs.recommender.cf.ALSRecommender
```
![als](https://mapr.com/blog/parallel-and-iterative-processing-machine-learning-recommendations-spark/assets/blogimages/SparkParallelIterativeBlog-Fig5.png)


- ItemCollaborationFilter-共现矩阵
```
com.lxs.recommender.cf.ItemCFRecommender
```
![itemcf](http://s3.51cto.com/wyfs02/M01/49/E9/wKiom1QfAT_xHw_EAAD7ZBEpZ4E049.jpg)

#### TB_TAG,用户的标签数据
- TagBased
    * K Means 聚类


// todo
k