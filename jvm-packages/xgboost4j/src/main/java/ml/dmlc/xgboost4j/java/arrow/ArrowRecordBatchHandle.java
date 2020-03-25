/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.xgboost4j.java.arrow;

import java.util.Arrays;
import java.util.List;

/**
 * Hold pointers to a Arrow C++ RecordBatch.
 * @see <a href="https://github.com/apache/arrow">Apache Arrow</a>
 * @see <a href="https://github.com/Intel-bigdata/arrow">Intel optimized Arrow</a>
 */
public class ArrowRecordBatchHandle {

  private final long numRows;
  private final Field[] fields;
  private final Buffer[] buffers;

  /**
   * Constructor.
   *
   * @param numRows Total row number of the associated RecordBatch
   * @param fields Metadata of fields
   * @param buffers Retained Arrow buffers
   */
  public ArrowRecordBatchHandle(long numRows, Field[] fields, Buffer[] buffers) {
    this.numRows = numRows;
    this.fields = fields;
    this.buffers = buffers;
  }

  /**
   * @return Total row number of the associated RecordBatch.
   */
  public long getNumRows() {
    return numRows;
  }

  /**
   * @return Metadata of fields.
   */
  public Field[] getFields() {
    return fields;
  }

  /**
   * @return Retained Arrow buffers.
   */
  public Buffer[] getBuffers() {
    return buffers;
  }

  /**
   * Field metadata.
   */
  public static class Field {
    private final long length;
    private final long nullCount;

    public Field(long length, long nullCount) {
      this.length = length;
      this.nullCount = nullCount;
    }

    public long getLength() {
      return length;
    }

    public long getNullCount() {
      return nullCount;
    }
  }

  /**
   * Pointers and metadata of the targeted Arrow buffer.
   */
  public static class Buffer {
    private final long memoryAddress;
    private final long size;
    private final long capacity;

    /**
     * Constructor.
     *
     * @param memoryAddress Memory address of the first byte
     * @param size Size (in bytes)
     * @param capacity Capacity (in bytes)
     */
    public Buffer(long memoryAddress, long size, long capacity) {
      this.memoryAddress = memoryAddress;
      this.size = size;
      this.capacity = capacity;
    }

    public long getMemoryAddress() {
      return memoryAddress;
    }

    public long getSize() {
      return size;
    }

    public long getCapacity() {
      return capacity;
    }
  }
}
