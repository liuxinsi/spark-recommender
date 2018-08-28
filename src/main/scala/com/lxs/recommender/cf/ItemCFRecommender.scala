package com.lxs.recommender.cf

import java.sql.DriverManager
import java.util.Properties

import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.Logger
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

/**
  * 共现相似。
  *
  * @author liuxinsi
  * @date 2018/8/28 15:41
  */
object ItemCFRecommender {
  private val log = Logger(ItemCFRecommender.getClass)

  /**
    * load config
    */
  private val config = ConfigFactory.load("config.properties")

  private val appName = config.getString("item.cf.appName")
  private val mode = config.getString("item.cf.mode")

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
      .set("spark.sql.codegen", "true")

    val session = SparkSession.builder().config(sc).getOrCreate()

    try {
      // 加载订单
      val orderDf = loadOrder(session)
      log.info("加载了:{}条订单数据", orderDf.count())


      // 保存推荐结果
      val itemPref = orderDf.rdd
        .map(f => ItemPref(f.getAs[String]("UID"), f.getAs[String]("PID"), f.getAs[Int]("ORDER_COUNT").toDouble))

      val simRdd = cooccurrenceSimi(itemPref)
      val rm = recommend(simRdd, itemPref, 10)

      val sored = rm.sortBy(f => f.pref, ascending = false)

      val rows = sored.map(f => Row(f.uid, f.pid, f.pref))

      dbInfo.put("createTableColumnTypes",
        "UID VARCHAR(200),PID VARCHAR(200),SCORE DOUBLE")
      session.createDataFrame(rows,
        StructType(
          StructField("UID", StringType) :: StructField("PID", StringType) :: StructField("SCORE", DoubleType) :: Nil
        ))
        .write
        .mode(SaveMode.Overwrite)
        .jdbc(dbUrl, "TB_RM_BASE_ON_USER_CF", dbInfo)

      createIndexes()
    } finally {
      session.close()
    }
  }

  /**
    * 创建索引
    */
  def createIndexes(): Unit = {
    Class.forName(dbDriver)
    val con = DriverManager.getConnection(dbUrl, dbUser, dbPwd)
    val statment = con.createStatement()
    try {
      statment.execute("CREATE INDEX TB_RM_BASE_ON_UCF_UID_IDX ON TB_RM_BASE_ON_USER_CF (UID)")
      log.debug("创建索引：TB_RM_BASE_ON_UCF_UID_IDX")

      statment.execute("CREATE INDEX TB_RM_BASE_ON_UCF_PID_IDX ON TB_RM_BASE_ON_USER_CF (PID)")
      log.debug("创建索引：TB_RM_BASE_ON_UCF_PID_IDX")
    } finally {
      statment.close()
      con.close()
    }
  }

  /**
    * @param ss spark
    * @return
    */
  def loadOrder(ss: SparkSession): DataFrame = {
    ss.read.jdbc(dbUrl, "TB_ORDER", dbInfo).cache()
  }

  /**
    * 用户评分。
    *
    * @param uid
    * @param pid
    * @param pref
    */
  case class ItemPref(uid: String, pid: String, pref: Double)

  /**
    * 用户推荐。
    *
    * @param uid
    * @param pid
    * @param pref
    */
  case class UserRm(uid: String, pid: String, pref: Double)

  /**
    * 相似度
    *
    * @param pid1
    * @param pid2
    * @param similar
    */
  case class ItemSimi(pid1: String, pid2: String, similar: Double)

  def cooccurrenceSimi(itemPref: RDD[ItemPref]): RDD[ItemSimi] = {
    val userRdd1 = itemPref.map(f => (f.uid, f.pid, f.pref))
    val userRdd2 = userRdd1.map(f => (f._1, f._2))

    // 笛卡尔积
    val userRdd3 = userRdd2.join(userRdd2)
    val userRdd4 = userRdd3.map(f => (f._2, 1))

    // 频率
    val userRdd5 = userRdd4.reduceByKey((x, y) => x + y)

    val userRdd6 = userRdd5.filter(f => f._1._1 == f._1._2)
    val userRdd7 = userRdd5.filter(f => f._1._1 != f._1._2)

    // 同现
    val userRdd8 = userRdd7.map(f => (f._1._1, (f._1._1, f._1._2, f._2))).join(userRdd6.map(f => (f._1._1, f._2)))
    val userRdd9 = userRdd8.map(f => (f._2._1._2, (f._2._1._1, f._2._1._2, f._2._1._3, f._2._2)))
    val userRdd10 = userRdd9.join(userRdd6.map(f => (f._1._1, f._2)))
    val userRdd11 = userRdd10.map(f => (f._2._1._1, f._2._1._2, f._2._1._3, f._2._1._4, f._2._2))
    val userRdd12 = userRdd11.map(f => (f._1, f._2, f._3 / math.sqrt(f._4 * f._5)))

    userRdd12.map(f => ItemSimi(f._1, f._2, f._3))
  }


  def recommend(itemSimi: RDD[ItemSimi], itemPref: RDD[ItemPref], num: Int): RDD[UserRm] = {
    val rddApp1R1 = itemSimi.map(f => (f.pid1, f.pid2, f.similar))
    val userPref1 = itemPref.map(f => (f.uid, f.pid, f.pref))

    val rddApp1R2 = rddApp1R1.map(f => (f._1, (f._2, f._3)))
      .join(userPref1.map(f => (f._2, (f._1, f._3))))

    val rddApp1R3 = rddApp1R2.map(f => ((f._2._2._1, f._2._1._1), f._2._2._2 * f._2._1._2))
    val rddApp1R4 = rddApp1R3.reduceByKey((x, y) => x + y)
    val rddApp1R5 = rddApp1R4.leftOuterJoin(userPref1.map(f => ((f._1, f._2), 1))).filter(f => f._2._2.isEmpty).map(f => (f._1._1, (f._1._2, f._2._1)))

    val rddAppR6 = rddApp1R5.groupByKey()
    val rddAppR7 = rddAppR6.map(f => {
      val i2 = f._2.toBuffer
      val i2_2 = i2.sortBy(_._2)
      if (i2_2.length > num) i2_2.remove(0, i2_2.length - num)
      (f._1, i2_2)
    })

    val rddAppR8 = rddAppR7.flatMap(f => {
      val id2 = f._2
      for (w <- id2) yield (f._1, w._1, w._2)
    })
    rddAppR8.map(f => UserRm(f._1, f._2, f._3))
  }
}
