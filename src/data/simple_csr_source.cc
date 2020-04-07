/*!
 * Copyright 2015-2019 by Contributors
 * \file simple_csr_source.cc
 */
#include <arrow/util/iterator.h>
#include <dmlc/base.h>
#include <xgboost/logging.h>
#include <xgboost/json.h>

#include <limits>
#include "simple_csr_source.h"
#include "columnar.h"

namespace xgboost {
namespace data {

void SimpleCSRSource::Clear() {
  page_.Clear();
  this->info.Clear();
}

void SimpleCSRSource::CopyFrom(DMatrix* src) {
  this->Clear();
  this->info = src->Info();
  for (const auto &batch : src->GetBatches<SparsePage>()) {
    page_.Push(batch);
  }
}

void SimpleCSRSource::CopyFrom(dmlc::Parser<uint32_t>* parser) {
  // use qid to get group info
  const uint64_t default_max = std::numeric_limits<uint64_t>::max();
  uint64_t last_group_id = default_max;
  bst_uint group_size = 0;
  std::vector<uint64_t> qids;
  this->Clear();
  while (parser->Next()) {
    const dmlc::RowBlock<uint32_t>& batch = parser->Value();
    if (batch.label != nullptr) {
      auto& labels = info.labels_.HostVector();
      labels.insert(labels.end(), batch.label, batch.label + batch.size);
    }
    if (batch.weight != nullptr) {
      auto& weights = info.weights_.HostVector();
      weights.insert(weights.end(), batch.weight, batch.weight + batch.size);
    }
    if (batch.qid != nullptr) {
      qids.insert(qids.end(), batch.qid, batch.qid + batch.size);
      // get group
      for (size_t i = 0; i < batch.size; ++i) {
        const uint64_t cur_group_id = batch.qid[i];
        if (last_group_id == default_max || last_group_id != cur_group_id) {
          info.group_ptr_.push_back(group_size);
        }
        last_group_id = cur_group_id;
        ++group_size;
      }
    }

    // Remove the assertion on batch.index, which can be null in the case that the data in this
    // batch is entirely sparse. Although it's true that this indicates a likely issue with the
    // user's data workflows, passing XGBoost entirely sparse data should not cause it to fail.
    // See https://github.com/dmlc/xgboost/issues/1827 for complete detail.
    // CHECK(batch.index != nullptr);

    // update information
    this->info.num_row_ += batch.size;
    // copy the data over
    auto& data_vec = page_.data.HostVector();
    auto& offset_vec = page_.offset.HostVector();
    for (size_t i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
      uint32_t index = batch.index[i];
      bst_float fvalue = batch.value == nullptr ? 1.0f : batch.value[i];
      data_vec.emplace_back(index, fvalue);
      this->info.num_col_ = std::max(this->info.num_col_,
                                    static_cast<uint64_t>(index + 1));
    }
    size_t top = page_.offset.Size();
    for (size_t i = 0; i < batch.size; ++i) {
      offset_vec.push_back(offset_vec[top - 1] + batch.offset[i + 1] - batch.offset[0]);
    }
  }
  if (last_group_id != default_max) {
    if (group_size > info.group_ptr_.back()) {
      info.group_ptr_.push_back(group_size);
    }
  }
  this->info.num_nonzero_ = static_cast<uint64_t>(page_.data.Size());
  // Either every row has query ID or none at all
  CHECK(qids.empty() || qids.size() == info.num_row_);
}

void CastAndFill(std::shared_ptr<arrow::BooleanArray> array, std::vector<xgboost::Entry>& vec,
    int64_t begin, unsigned int offset, int stride) {
  for (int64_t i = 0; i < array->length(); ++i) {
    vec[begin + i * stride + offset] =
      xgboost::Entry{offset, static_cast<bst_float>(array->Value(i))};
  }
}
    
template <typename T,
         typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
void CastAndFill(std::shared_ptr<arrow::BooleanArray> array, std::vector<T>& vec,
    int64_t begin, unsigned int offset, int stride) {
  for (int64_t i = 0; i < array->length(); ++i) {
    vec[begin + i * stride + offset] = static_cast<T>(array->Value(i));
  }
}

template <typename ArrayType>
void CastAndFill(std::shared_ptr<ArrayType> array, std::vector<xgboost::Entry>& vec,
    int64_t begin, unsigned int offset, int stride) {
  const auto* raw = array->raw_values();
  for (int64_t i = 0; i < array->length(); ++i) {
    vec[begin + i * stride + offset] = xgboost::Entry{offset, static_cast<bst_float>(raw[i])};
  }
}

template <typename ArrayType, typename T,
         typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
void CastAndFill(std::shared_ptr<ArrayType> array, std::vector<T>& vec,
    int64_t begin, unsigned int offset, int stride) {
  const auto* raw = array->raw_values();
  for (int64_t i = 0; i < array->length(); ++i) {
    vec[begin + i * stride + offset] = static_cast<T>(raw[i]);
  }
}

template <typename T>
void CastArrowArray(std::shared_ptr<arrow::Array> array, std::vector<T>& vec,
    int64_t begin, unsigned int offset, int stride) {
  switch (array->type_id()) {
    case arrow::Type::type::BOOL:
      CastAndFill(std::static_pointer_cast<arrow::BooleanArray>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::UINT8:
      CastAndFill(std::static_pointer_cast<arrow::UInt8Array>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::INT8:
      CastAndFill(std::static_pointer_cast<arrow::Int8Array>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::UINT16:
      CastAndFill(std::static_pointer_cast<arrow::UInt16Array>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::INT16:
      CastAndFill(std::static_pointer_cast<arrow::Int16Array>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::UINT32:
      CastAndFill(std::static_pointer_cast<arrow::UInt32Array>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::INT32:
      CastAndFill(std::static_pointer_cast<arrow::Int32Array>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::UINT64:
      CastAndFill(std::static_pointer_cast<arrow::UInt64Array>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::INT64:
      CastAndFill(std::static_pointer_cast<arrow::Int64Array>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::FLOAT:
      CastAndFill(std::static_pointer_cast<arrow::FloatArray>(array), vec, begin, offset, stride);
      break;
    case arrow::Type::type::DOUBLE:
      CastAndFill(std::static_pointer_cast<arrow::DoubleArray>(array), vec, begin, offset, stride);
      break;
  }
}

void SimpleCSRSource::CopyFrom(arrow::RecordBatchIterator& batches, std::string label) {
  this->Clear();
  for (auto result : batches) {
    if (result.ok()) {
      const std::shared_ptr<arrow::RecordBatch>& sp_batch = result.ValueOrDie();
      std::cout << "sp_batch->schema(): " << sp_batch->schema().get() << "\n" << std::flush;
      for (std::string name : sp_batch->schema()->field_names()) {
        std::cout << "name: " << name << "\n" << std::flush;
      }
      std::cout << "label: " << label << "\n" << std::flush;
      std::cout << "sp_batch-> num_columns(): " << sp_batch->num_columns() << "\n" << std::flush;
      std::cout << "sp_batch-> num_rows(): " << sp_batch->num_rows() << "\n" << std::flush;
      std::cout << "sp_batch->GetColumnByName(label): " << sp_batch->GetColumnByName(label).get() << "\n" << std::flush;
      std::cout << "sp_batch->column(0): " << sp_batch->column(0).get() << "\n" << std::flush;
      std::cout << "sp_batch->GetColumnByName(label)->length(): " << sp_batch->GetColumnByName(label)->length() << "\n" << std::flush;
      std::cout << "sp_batch->column(0)->length(): " << sp_batch->column(0)->length() << "\n" << std::flush;
      // num_row_
      int64_t batch_num_rows = sp_batch->num_rows();
      info.num_row_ += batch_num_rows;
      // num_col_
      int batch_num_cols = sp_batch->num_columns();
      CHECK(info.num_col_ == 0 || info.num_col_ == batch_num_cols - 1);
      if (info.num_col_ == 0) {
        // exclude the label column
        info.num_col_ = batch_num_cols - 1;
      }
      // label
      auto label_col = sp_batch->GetColumnByName(std::move(label));
      auto& labels = info.labels_.HostVector();
      int64_t begin = info.labels_.Size();
      labels.resize(begin + batch_num_rows);
      CastArrowArray(label_col, labels, begin, 0, 1);
      // data
      auto& data_vec = page_.data.HostVector();
      begin = page_.data.Size();
      data_vec.resize(begin + batch_num_rows * info.num_col_);
      unsigned int offset = 0;
      for (int j = 0; j < batch_num_cols; ++j) {
        if (sp_batch->column_name(j) != label) {
          CastArrowArray(sp_batch->column(j), data_vec, begin, offset++, info.num_col_);
        }
      }
      // offset
      size_t top = page_.offset.Size();
      auto& offset_vec = page_.offset.HostVector();
      for (int64_t i = 0; i < batch_num_rows; ++i) {
        offset_vec.push_back(offset_vec[top - 1] + (i + 1) * info.num_col_);
      }
    }
  }
  // num_nonzero_
  info.num_nonzero_ = static_cast<uint64_t>(page_.data.Size());
}

void SimpleCSRSource::LoadBinary(dmlc::Stream* fi) {
  int tmagic;
  CHECK(fi->Read(&tmagic, sizeof(tmagic)) == sizeof(tmagic)) << "invalid input file format";
  CHECK_EQ(tmagic, kMagic) << "invalid format, magic number mismatch";
  info.LoadBinary(fi);
  fi->Read(&page_.offset.HostVector());
  fi->Read(&page_.data.HostVector());
}

void SimpleCSRSource::SaveBinary(dmlc::Stream* fo) const {
  int tmagic = kMagic;
  fo->Write(&tmagic, sizeof(tmagic));
  info.SaveBinary(fo);
  fo->Write(page_.offset.HostVector());
  fo->Write(page_.data.HostVector());
}

void SimpleCSRSource::BeforeFirst() {
  at_first_ = true;
}

bool SimpleCSRSource::Next() {
  if (!at_first_) return false;
  at_first_ = false;
  return true;
}

const SparsePage& SimpleCSRSource::Value() const {
  return page_;
}

/*!
 * Please be careful that, in official specification, the only three required fields are
 * `shape', `version' and `typestr'.  Any other is optional, including `data'.  But here
 * we have one additional requirements for input data:
 *
 * - `data' field is required, passing in an empty dataset is not accepted, as most (if
 *   not all) of our algorithms don't have test for empty dataset.  An error is better
 *   than a crash.
 *
 * Missing value handling:
 *   Missing value is specified:
 *     - Ignore the validity mask from columnar format.
 *     - Remove entries that equals to missing value.
 *     - missing = NaN:
 *        - Remove entries that is NaN
 *     - missing != NaN:
 *        - Check for NaN entries, throw an error if found.
 *   Missing value is not specified:
 *     - Remove entries that is specifed as by validity mask.
 *     - Remove NaN entries.
 *
 * What if invalid value from dataframe is 0 but I specify missing=NaN in XGBoost?  Since
 * validity mask is ignored, all 0s are preserved in XGBoost.
 *
 * FIXME(trivialfis): Put above into document after we have a consistent way for
 * processing input data.
 *
 * Sample input:
 * [
 *   {
 *     "shape": [
 *       10
 *     ],
 *     "strides": [
 *       4
 *     ],
 *     "data": [
 *       30074864128,
 *       false
 *     ],
 *     "typestr": "<f4",
 *     "version": 1,
 *     "mask": {
 *       "shape": [
 *         64
 *       ],
 *       "strides": [
 *         1
 *       ],
 *       "data": [
 *         30074864640,
 *         false
 *       ],
 *       "typestr": "|i1",
 *       "version": 1
 *     }
 *   }
 * ]
 */
void SimpleCSRSource::CopyFrom(std::string const& cuda_interfaces_str,
                               bool has_missing, float missing) {
  Json interfaces = Json::Load({cuda_interfaces_str.c_str(),
                                cuda_interfaces_str.size()});
  std::vector<Json> const& columns = get<Array>(interfaces);
  size_t n_columns = columns.size();
  CHECK_GT(n_columns, 0) << "Number of columns must not eqaul to 0.";

  auto const& typestr = get<String const>(columns[0]["typestr"]);
  CHECK_EQ(typestr.size(),    3)  << ColumnarErrors::TypestrFormat();
  CHECK_NE(typestr.front(), '>')  << ColumnarErrors::BigEndian();

  this->FromDeviceColumnar(columns, has_missing, missing);
}

#if !defined(XGBOOST_USE_CUDA)
void SimpleCSRSource::FromDeviceColumnar(std::vector<Json> const& columns,
                                         bool has_missing, float missing) {
  LOG(FATAL) << "XGBoost version is not compiled with GPU support";
}
#endif  // !defined(XGBOOST_USE_CUDA)

}  // namespace data
}  // namespace xgboost
