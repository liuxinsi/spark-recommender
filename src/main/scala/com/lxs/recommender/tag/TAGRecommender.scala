package com.lxs.recommender.tag

import java.sql.DriverManager
import java.util.Properties
import java.util
import java.util.function.Consumer

import com.google.common.collect.HashMultimap
import org.apache.spark.SparkConf
import org.apache.spark.api.java.function.{FlatMapFunction, MapFunction}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.{ExpressionEncoder, RowEncoder}
import org.apache.spark.sql.types._

import scala.collection.{JavaConversions, mutable}
import scala.collection.mutable.ListBuffer
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.Logger
import org.apache.spark.broadcast.Broadcast

/**
  * 根据标签聚类，同组用户相互推荐。
  *
  * @author liuxinsi
  * @date 2018/8/28 16:15
  */
object TAGRecommender {
  private val log = Logger(TAGRecommender.getClass)

  /**
    * load config
    */
  private val config = ConfigFactory.load("config.properties")

  private val appName = config.getString("tag.appName")
  private val mode = config.getString("tag.mode")

  private val dbUser = config.getString("dbUser")
  private val dbPwd = config.getString("dbPwd")
  private val dbDriver = config.getString("dbDriver")

  /**
    * db
    */
  val dbUrl: String = config.getString("dbUrl")
  var dbInfo: Properties = {
    dbInfo = new Properties()
    dbInfo.put("user", dbUser)
    dbInfo.put("password", dbPwd)
    dbInfo.put("driver", dbDriver)
    dbInfo
  }

  def main(args: Array[String]): Unit = {
    log.info("init {}，mode:{}", appName, mode)

    // init spark
    val sc = new SparkConf()
      .setAppName(appName)
      .setMaster(mode)
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val session = SparkSession.builder().config(sc).getOrCreate()
    try {
      // 矩阵
      val mat = initMat(session)

      // 聚类
      cluster(mat, session)

      // 加载用户分组信息
      // k=组id,v=用户列表
      val userGroupDf = loadUserGroup(session)

      // 加载用户标签
      // k=uid,v=标签列表
      val userTagMap = loadUserTags(session)
      val tagMapBc = session.sparkContext.broadcast[mutable.HashMap[String, List[String]]](userTagMap)

      // 加载订单信息
      // k=uid,v=订单信息
      val orderMap = loadOrder(session)
      val orderMapBc = session.sparkContext.broadcast[mutable.HashMap[String, ListBuffer[Order]]](orderMap)

      val schema = StructType(Seq(
        StructField("UID", StringType),
        StructField("PID", StringType),
        StructField("SCORE", DoubleType)
      ))

      val encoder = RowEncoder(schema)
      val rmDf = userGroupDf.flatMap(new FlatMapFunction[Row, Row] {
        override def call(r: Row): util.Iterator[Row] = {
          // 组
          val group = r.getAs[Int]("GROUP")
          // 组下用户列表
          val uids = r.getAs[mutable.WrappedArray.ofRef[String]]("UIDS")
          val userList = uids.toList

          // 标签完全一致的用户
          val sameTagsUser = HashMultimap.create[String, String]()
          userList.foreach(uid => {
            val userTags = tagMapBc.value(uid)
            sameTagsUser.put(userTags.mkString("-"), uid)
          })

          val keys = sameTagsUser.keySet()

          val rowList = new ListBuffer[Row]
          keys.forEach(new Consumer[String] {
            override def accept(k: String): Unit = {
              val userGroup = sameTagsUser.get(k)
              val firstUid = userGroup.iterator().next()

              val rows = getRecomends(firstUid, sameTagsUser.values(), tagMapBc, orderMapBc)
              rowList ++= rows

              userGroup.forEach(new Consumer[String] {
                override def accept(t: String): Unit = {
                  if (!t.equals(firstUid)) {
                    rows.foreach(r => {
                      rowList += Row(t, r.get(1), r.get(2))
                    })
                  }
                }
              })
            }
          })

          JavaConversions.asJavaIterator(rowList.iterator)
        }
      }, encoder)

      //
      saveRm(rmDf)
      createIndexes()
    } finally {
      session.close()
    }
  }

  /**
    * 聚类
    *
    * @param den 用户标签矩阵
    * @param ss
    */
  def cluster(den: Dataset[List[String]], ss: SparkSession): Unit = {
    val mat = den.toJavaRDD.rdd.map(v => {
      Vectors.dense(v.slice(1, v.size).map(_.toDouble).toArray)
    }).cache()

    val karr = List(5, 10, 15, 20)
    val iterArr = List(50, 100, 200, 300, 500, 800, 1000)

    val result = new util.ArrayList[Row]()
    for (k <- karr; it <- iterArr) {
      val model = KMeans.train(mat, k, it)
      //
      val kMeansCost = model.computeCost(mat)

      result.add(Row(k, it, kMeansCost))
    }
    ss.createDataFrame(result,
      StructType(StructField("K", IntegerType) ::
        StructField("MAX_ITERATIONS", IntegerType) ::
        StructField("COST", DoubleType) :: Nil)
    )
      .write
      .mode(SaveMode.Overwrite)
      .jdbc(dbUrl, "TB_K_PARAMS", dbInfo)

    val trainParamsRdd = ss.read.jdbc(dbUrl, "TB_K_PARAMS", dbInfo)

    val paramRow = trainParamsRdd.sort("COST").first()
    val k = paramRow.getAs[Int]("K")
    val itera = paramRow.getAs[Int]("MAX_ITERATIONS")
    log.info("选择了k:" + k + ",it:" + itera)
    val model = KMeans.train(mat, k, itera)

    // 分组用户
    val gpusers = new util.ArrayList[Row]()
    den.toLocalIterator().forEachRemaining(new Consumer[List[String]] {
      override def accept(v: List[String]): Unit = {
        val sim = model.predict(Vectors.dense(v.slice(1, v.size).map(_.toDouble).toArray))
        val uid = v.slice(0, 1).head
        gpusers.add(Row(sim, uid))
      }
    })

    ss.createDataFrame(gpusers, StructType(
      StructField("GROUP", IntegerType) :: StructField("UID", StringType) :: Nil
    )).write.mode(SaveMode.Overwrite).jdbc(dbUrl, "TB_USER_GROUP", dbInfo)
  }

  /**
    * 生成矩阵
    *
    * @param ss
    */
  def initMat(ss: SparkSession): Dataset[List[String]] = {
    // 加载用户和标签
    val df = ss.read.jdbc(dbUrl, "TB_USER_TAG", dbInfo).cache()

    implicit val stringEncoder: Encoder[String] = org.apache.spark.sql.Encoders.STRING
    val tagDf = df.map(row => row.getAs[String]("TAG")).distinct()

    // 生成矩阵
    var headMap = new mutable.LinkedHashMap[String, Int]()
    val heads = new ListBuffer[String]

    heads += "UID"
    headMap.put("UID", 0)

    tagDf.toLocalIterator().forEachRemaining(new Consumer[String] {
      def accept(t: String): Unit = {
        heads += t
        headMap.put(t, heads.indexOf(t))
      }
    })

    val collectDf = df.groupBy("UID")
      .agg(
        functions.collect_set("TAG").alias("TAG"),
        functions.collect_set("ID").alias("ID")
      )

    implicit def mapIntIntEncoder: Encoder[List[String]] = ExpressionEncoder()

    val headBroadcast = ss.sparkContext.broadcast[mutable.LinkedHashMap[String, Int]](headMap)

    // 矩阵：
    // uid,tag1,tag2,tag3,...
    // u1,1,0,0,...
    // u2,0,1,1,...
    val den = collectDf.map[List[String]](new MapFunction[Row, List[String]] {
      def call(row: Row): List[String] = {
        val list = new ListBuffer[String]

        val uid = row.getAs[String]("UID")
        val ids = row.getAs[mutable.WrappedArray.ofRef[String]]("ID")
        val tags = row.getAs[mutable.WrappedArray.ofRef[String]]("TAG")

        headBroadcast.value.keySet.foreach(key => {
          if (key.equals("UID")) {
            list += uid
          } else {
            if (tags.toList.contains(key)) {
              list += "1"
            } else {
              list += "0"
            }
          }
        })

        list.toList
      }
    }, mapIntIntEncoder)
      .cache()
    den
  }

  case class Order(uid: String, pid: String, orderCount: Int)

  def getRecomends(uid: String,
                   userList: util.Collection[String],
                   tagMapBc: Broadcast[mutable.HashMap[String, List[String]]],
                   orderMapBc: Broadcast[mutable.HashMap[String, ListBuffer[Order]]]
                  ): List[Row] = {
    // 获取用户标签
    val userTags = tagMapBc.value(uid)

    // 获取该组下其他用户
    val sameUsers = JavaConversions.asScalaIterator(userList.iterator()).filter(u => u != uid)

    // 保存其他用户标签数量
    // k=uid,v=一样标签的数量
    val tagSizeMap = new mutable.HashMap[String, Int]()
    sameUsers.foreach(sameUserId => {
      val sameUserTags = tagMapBc.value(sameUserId)

      // 交集标签数量
      val size = sameUserTags.intersect(userTags).size
      tagSizeMap.put(sameUserId, size)
    })

    // 按标签数量排序
    var sortedMap = mutable.LinkedHashMap(
      tagSizeMap.toSeq.sortWith((u1, u2) => {
        // 如标签数量一样
        if (u1._2 == u2._2) {
          // 购买数量多的排前
          var u1OrderSize = 0
          var u2OrderSize = 0
          if (orderMapBc.value.contains(u1._1)) {
            u1OrderSize = orderMapBc.value(u1._1).toList.size
          }

          if (orderMapBc.value.contains(u2._1)) {
            u2OrderSize = orderMapBc.value(u2._1).toList.size
          }
          u1OrderSize > u2OrderSize
        } else {
          // 标签数越多的排前
          u1._2 > u2._2
        }
      }): _*)

    // top 20
    if (sortedMap.size > 20) {
      sortedMap = sortedMap.take(20)
    }

    // 推荐
    // k=pid,v=推荐度
    val recommendMap = new mutable.HashMap[String, Double]()
    sortedMap.foreach(f => {
      val uid = f._1

      // 如该用户有订单
      if (orderMapBc.value.contains(uid)) {
        val orders = orderMapBc.value(uid).toList

        orders.foreach(o => {
          if (recommendMap.contains(o.pid)) {
            recommendMap.put(o.pid, recommendMap(o.pid) + 1)
          } else {
            recommendMap.put(o.pid, 1)
          }
        })
      }
    })

    // 排序
    var sortedRmMap = mutable.LinkedHashMap(recommendMap.toSeq.sortWith(_._2 > _._2): _*)

    if (sortedRmMap.size > 10) {
      sortedRmMap = sortedRmMap.take(10)
    }

    sortedRmMap.map(rm => Row(uid, rm._1, rm._2)).toList
  }

  def saveRm(rmDf: DataFrame): Unit = {
    dbInfo.put("createTableColumnTypes",
      "UID VARCHAR(200),PID VARCHAR(200),SCORE DOUBLE")
    rmDf.write
      .mode(SaveMode.Overwrite)
      .jdbc(dbUrl, "TB_RM_BASE_ON_TAG", dbInfo)
  }

  /**
    * 创建索引
    */
  def createIndexes(): Unit = {
    Class.forName(dbDriver)
    val con = DriverManager.getConnection(dbUrl, dbUser, dbPwd)
    val statment = con.createStatement()
    try {
      statment.execute("CREATE INDEX TB_RM_BASE_ON_TAG_UID_IDX ON TB_RM_BASE_ON_TAG (UID)")
      log.debug("创建索引：TB_RM_BASE_ON_TAG_UID_IDX")

      statment.execute("CREATE INDEX TB_RM_BASE_ON_TAG_PID_IDX ON TB_RM_BASE_ON_TAG (PID)")
      log.debug("创建索引：TB_RM_BASE_ON_TAG_PID_IDX")
    } finally {
      statment.close()
      con.close()
    }
  }

  /**
    * 加载用户订单。
    *
    * @param ss
    * @return k=uid，v=该用户的订单
    */
  def loadOrder(ss: SparkSession): mutable.HashMap[String, ListBuffer[Order]] = {
    import ss.implicits._
    val orderDf = ss.read
      .jdbc(dbUrl, "TB_ORDER", dbInfo)
      .cache()
      .map(r => Order(r.getAs[String]("UID"), r.getAs[String]("PID"), r.getAs[Int]("ORDER_COUNT")))

    val orderMap = new mutable.HashMap[String, ListBuffer[Order]]()
    orderDf.toLocalIterator().forEachRemaining(new Consumer[Order] {
      override def accept(r: Order): Unit = {
        if (orderMap.contains(r.uid)) {
          orderMap(r.uid) += r
        } else {
          val list = new ListBuffer[Order]
          list += r
          orderMap.put(r.uid, list)
        }
      }
    })
    orderMap
  }

  /**
    * 加载用户分组。
    *
    * @param ss
    * @return
    */
  def loadUserGroup(ss: SparkSession): DataFrame = {
    ss.read
      .jdbc(dbUrl, "TB_USER_GROUP", dbInfo)
      .groupBy("GROUP")
      .agg(functions.collect_list("UID").as("UIDS"))
      .cache()
  }

  /**
    * 加载用户标签。
    *
    * @param ss
    * @return k=uid,v=用户拥有的标签
    */
  def loadUserTags(ss: SparkSession): mutable.HashMap[String, List[String]] = {
    val userTagDf = ss.read
      .jdbc(dbUrl, "TB_USER_TAG", dbInfo)
      .groupBy("UID")
      .agg(functions.collect_list("TAG").as("TAGS"))
      .cache()

    val tagMap = new mutable.HashMap[String, List[String]]()
    userTagDf.toLocalIterator().forEachRemaining(new Consumer[Row] {
      override def accept(r: Row): Unit = {
        val uid = r.getAs[String]("UID")
        val tags = r.getAs[mutable.WrappedArray.ofRef[String]]("TAGS")
        tagMap.put(uid, tags.toList)
      }
    })
    tagMap
  }
}
