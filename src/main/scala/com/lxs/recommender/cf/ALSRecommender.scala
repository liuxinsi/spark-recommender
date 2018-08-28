package com.lxs.recommender.cf

import java.sql.DriverManager
import java.util.Properties
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.Logger
import org.apache.spark.SparkConf
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types._
import scala.collection.mutable

/**
  * 矩阵分解。
  *
  * @author liuxinsi
  * @date 2018/8/28 14:57
  */
object ALSRecommender {
  private val log = Logger(ALSRecommender.getClass)

  /**
    * load config
    */
  private val config = ConfigFactory.load("config.properties")

  private val appName = config.getString("als.appName")
  private val mode = config.getString("als.mode")
  private val imp = config.getBoolean("als.implicit")
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
    log.info("init {}，mode:{}，implicit:{}", appName, mode, imp)

    // init spark
    val sc = new SparkConf()
      .setAppName(appName)
      .setMaster(mode)
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.codegen", "true")

    val session = SparkSession.builder().config(sc).getOrCreate()
    session.sparkContext.setCheckpointDir(config.getString("als.checkpoint"))

    try {
      // 加载订单
      val orderDf = loadOrder(session)
      log.info("加载了:{}条订单数据", orderDf.count())

      // 为用户映射唯一id
      val userTuple = userMapping(orderDf)
      log.info("{}条用户信息", userTuple._1.size)

      // 为产品映射唯一id
      val productTuple = productMapping(orderDf)
      log.info("{}条产品信息", productTuple._1.size)

      // broadcast
      val userBroadcast = session.sparkContext.broadcast[collection.Map[String, Long]](userTuple._1)
      val inverseUserBroadcast = session.sparkContext.broadcast[collection.Map[Long, String]](userTuple._2)
      val productBroadcast = session.sparkContext.broadcast[collection.Map[String, Long]](productTuple._1)
      val inverseProductBroadcast = session.sparkContext.broadcast[collection.Map[Long, String]](productTuple._2)

      // rating
      val ratings = buildRating(orderDf, userBroadcast, productBroadcast)
      val splitedRats = ratings.randomSplit(Array(0.6, 0.2, 0.2))
      val train = splitedRats(0)
      val test = splitedRats(1)
      val validate = splitedRats(2)

      log.info("{}条rating，训练集：{}，测试集：{}，验证集：{}",
        ratings.count(), train.count(), test.count(), validate.count())

      // train
      val ranks = List(10, 20, 30, 40, 50, 60)
      val numIterations = List(10, 20, 30, 40, 50, 60)

      var bestMes = Double.MaxValue
      var bestRank = 0
      var bestNumIt = 0
      var bestModel: MatrixFactorizationModel = null
      for (rank <- ranks; it <- numIterations) {

        var model = None: Option[MatrixFactorizationModel]
        if (imp) {
          model = Some(ALS.trainImplicit(train, rank, it))
        } else {
          model = Some(ALS.train(train, rank, it))
        }

        // 均方差
        val mse = getMse(validate, model.get)
        if (mse < bestMes) {
          bestMes = mse
          bestRank = rank
          bestNumIt = it
          bestModel = model.get
        }
      }
      log.debug("最佳模型，MES：{}，Rank：{}，iterator：{}", bestMes, bestRank, bestNumIt)

      // test
      val testMsg = getMse(test, bestModel)
      val row = Row(train.count(), validate.count(), test.count(), bestRank, bestNumIt, bestMes, testMsg)

      // 保存model
      session.createDataFrame(session.sparkContext.makeRDD(Seq(row)),
        StructType(StructField("TRAIN_SIZE", LongType) ::
          StructField("VALIDATION_SIZE", LongType) ::
          StructField("TEST_SIZE", LongType) ::
          StructField("RANK", IntegerType) ::
          StructField("ITERATIONS", IntegerType) ::
          StructField("VALIDATION_MSE", DoubleType) ::
          StructField("TEST_MSE", DoubleType) :: Nil)
      ).write
        .mode(SaveMode.Overwrite)
        .jdbc(dbUrl, "TB_CF_TRAIN_MODEL", dbInfo)

      // 保存推荐结果
      save(bestModel, session, inverseUserBroadcast, inverseProductBroadcast)

      createIndexes()
    } finally {
      session.close()
    }
  }


  /**
    * 保存推荐结果。
    *
    * @param model                   推荐结果
    * @param session                 spark
    * @param inverseUserBroadcast    k=uid映射的long值,v=uid
    * @param inverseProductBroadcast k=pid映射的long值,v=pid
    */
  def save(model: MatrixFactorizationModel, session: SparkSession,
           inverseUserBroadcast: Broadcast[collection.Map[Long, String]],
           inverseProductBroadcast: Broadcast[collection.Map[Long, String]]): Unit = {
    val rows = model.recommendProductsForUsers(10).cache().flatMap[Row](f => {
      // 推荐结果映射
      f._2.map(r => Row(
        inverseUserBroadcast.value(r.user.longValue()),
        inverseProductBroadcast.value(r.product.longValue()),
        r.rating
      ))
    })

    // 构造表结构
    dbInfo.put("createTableColumnTypes",
      "UID VARCHAR(200),PID VARCHAR(200),SCORE DOUBLE")
    session.createDataFrame(rows,
      StructType(
        StructField("UID", StringType) :: StructField("PID", StringType) :: StructField("SCORE", DoubleType) :: Nil
      ))
      .write
      .mode(SaveMode.Overwrite)
      .jdbc(dbUrl, "TB_RM_BASE_ON_ALS", dbInfo)
  }

  /**
    * 创建索引
    */
  def createIndexes(): Unit = {
    Class.forName(dbDriver)
    val con = DriverManager.getConnection(dbUrl, dbUser, dbPwd)
    val statment = con.createStatement()
    try {
      statment.execute("CREATE INDEX TB_RM_BASE_ON_ALS_UID_IDX ON TB_RM_BASE_ON_ALS (UID)")
      log.debug("创建索引：TB_RM_BASE_ON_ALS_UID_IDX")

      statment.execute("CREATE INDEX TB_RM_BASE_ON_ALS_PID_IDX ON TB_RM_BASE_ON_ALS (PID)")
      log.debug("创建索引：TB_RM_BASE_ON_ALS_PID_IDX")
    } finally {
      statment.close()
      con.close()
    }
  }

  /**
    * 计算均方差。
    *
    * @param ratings
    * @param model
    * @return
    */
  def getMse(ratings: RDD[Rating], model: MatrixFactorizationModel): Double = {
    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }

    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions)

    ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()
  }

  /**
    * @param ss spark
    * @return
    */
  def loadOrder(ss: SparkSession): DataFrame = {
    ss.read.jdbc(dbUrl, "TB_ORDER", dbInfo).cache()
  }

  /**
    * 将uid映射一个唯一的long。<br/>
    * tuple_1=map,k=uid,v=uid对应的long。<br/>
    * tuple_2=inverse map,k=uid对应的long,v=uid。<br/>
    *
    * @param df 订单数据
    * @return
    */
  def userMapping(df: DataFrame): Tuple2[collection.Map[String, Long], collection.Map[Long, String]] = {
    val userMap = df.rdd
      .map[String](r => r.getAs[String]("UID"))
      .distinct()
      .zipWithUniqueId()
      .collectAsMap()

    val inverseUserMap = new mutable.HashMap[Long, String]()
    userMap.foreach(f => inverseUserMap.put(f._2, f._1))

    Tuple2(userMap, inverseUserMap)
  }

  /**
    * 将pid映射一个唯一的long。<br/>
    * tuple_1=map,k=pid,v=pid对应的long。<br/>
    * tuple_2=inverse map,k=pid对应的long,p=uid。<br/>
    *
    * @param df 订单数据
    * @return
    */
  def productMapping(df: DataFrame): Tuple2[collection.Map[String, Long], collection.Map[Long, String]] = {
    val productMap = df.rdd
      .map[String](r => r.getAs[String]("PID"))
      .distinct()
      .zipWithUniqueId()
      .collectAsMap()

    val inverseProductMap = new mutable.HashMap[Long, String]()
    productMap.foreach(f => inverseProductMap.put(f._2, f._1))

    Tuple2(productMap, inverseProductMap)
  }

  /**
    * 构建推荐计算对象[[Rating]]。
    *
    * @param df        订单数据
    * @param userBc    k=uid,v=uid对应的long
    * @param productBc k=pid,v=pid对应的long
    * @return
    */
  def buildRating(df: DataFrame, userBc: Broadcast[collection.Map[String, Long]],
                  productBc: Broadcast[collection.Map[String, Long]]): RDD[Rating] = {
    implicit def ratEncoder: Encoder[Rating] = ExpressionEncoder()

    df.map(new MapFunction[Row, Rating]() {
      def call(r: Row): Rating = {
        val uid = r.getAs[String]("UID")
        val pid = r.getAs[String]("PID")
        val count = r.getAs[Int]("ORDER_COUNT")

        Rating(userBc.value(uid).intValue(), productBc.value(pid).intValue(), count.toFloat)
      }
    }, ratEncoder).rdd.cache()
  }
}


