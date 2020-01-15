/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package org.apache.spark.rdd

import org.apache.spark.Partition
import org.apache.spark.scheduler.ExecutorCacheTaskLocation

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class ExecutorInProcessCoalescePartitioner(val balanceSlack: Double = 0.10)
  extends PartitionCoalescer with Serializable {
  def coalesce(maxPartitions: Int, prev: RDD[_]): Array[PartitionGroup] = {
    val map = new mutable.HashMap[String, mutable.HashSet[Partition]]()
    val groupArr = ArrayBuffer[PartitionGroup]()
    prev.partitions.foreach(p => {
      val loc = prev.context.getPreferredLocs(prev, p.index)
      loc.foreach{
      case location : ExecutorCacheTaskLocation =>
        val execLoc = "executor_" + location.host + "_" + location.executorId
        val partValue = map.getOrElse(execLoc, new mutable.HashSet[Partition]())
        partValue.add(p)
        map.put(execLoc, partValue)
      case _ =>
          // skip if not executor cache location
      }
    })
      map.foreach(x => {
      val pg = new PartitionGroup(Some(x._1))
      x._2.foreach(part => pg.partitions += part)
      groupArr += pg
    })
    return groupArr.toArray
  }
}
