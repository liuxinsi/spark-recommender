# spark-recommender
使用Spark实现一些常用的推荐算法。

### 样本数据data.db
#### TB_ORDER,用户对商品的购买记录30w

UID | PID | ORDER_COUNT|
| :------: | :------: | :------: |
用户id | 商品id | 购买数量|

<br>

- ALS矩阵分解
```
com.lxs.recommender.cf.ALSRecommender
```
![als](https://mapr.com/blog/parallel-and-iterative-processing-machine-learning-recommendations-spark/assets/blogimages/SparkParallelIterativeBlog-Fig5.png)

<br>

- ItemCollaborationFilter-共现矩阵
```
com.lxs.recommender.cf.ItemCFRecommender
```
![itemcf](http://s3.51cto.com/wyfs02/M01/49/E9/wKiom1QfAT_xHw_EAAD7ZBEpZ4E049.jpg)

<br>
<br>

#### TB_TAG,用户的标签数据
UID | TAG |
| :------: | :------: |
用户id | 用户标签 |

<br>

- TagBased
    * K Means 聚类
    * 互推
```
com.lxs.recommender.tag.TAGRecommender
```
![tag](https://image.slidesharecdn.com/abbasi-samt-2009-091203084232-phpapp01/95/large-scale-tag-recommendation-using-different-image-representations-10-728.jpg?cb=1259829801)
<br>
<br>

### Usage
- 开发模式下Maven Profile使用dev模式，各个脚本中有入口Main.
- 集群模式下Maven Profile使用production模式。
    - maven assembly -P production,als,!dev
    - maven assembly -P production,icf,!dev
    - maven assembly -P production,tag,!dev
