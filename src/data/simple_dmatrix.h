/*!
 * Copyright 2015 by Contributors
 * \file simple_dmatrix.h
 * \brief In-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_H_
#define XGBOOST_DATA_SIMPLE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>
#include <forward_list>
#include <mutex>
#include <condition_variable>

#include "simple_csr_source.h"

namespace xgboost {
namespace data {
// Used for single batch data.
class SimpleDMatrix : public DMatrix {
 public:
  explicit SimpleDMatrix(std::unique_ptr<DataSource<SparsePage>>&& source)
      : source_(std::move(source)) {}

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  float GetColDensity(size_t cidx) override;

  bool SingleColBlock() const override;

  std::vector<Entry> GetColumn(size_t idx) const override {
    std::vector<Entry> ret;
    auto page = dynamic_cast<SimpleCSRSource*>(source_.get())->page_;
    const auto& data = page.data.HostVector();
    int num_cols = Info().num_col_;
    for (size_t i = 0; i < Info().num_row_; ++i) {
      ret.push_back(data[i * num_cols + idx]);
    }
    return ret;
  }

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;

  // source data pointer.
  std::unique_ptr<DataSource<SparsePage>> source_;

  std::unique_ptr<CSCPage> column_page_;
  std::unique_ptr<SortedCSCPage> sorted_column_page_;
  std::unique_ptr<EllpackPage> ellpack_page_;
};

// Used for multi-batch data.
class BatchedDMatrix : public DMatrix {
 public:
  static BatchedDMatrix* getBatchedDMatrix();

  bool AddBatch(std::unique_ptr<SimpleCSRSource>&& batch);

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  float GetColDensity(size_t cidx) override;

  bool SingleColBlock() const override;

  // for testing
  size_t GetNumRows();
  std::vector<Entry> GetColumn(size_t idx) const override;

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;
  explicit BatchedDMatrix() : nBatches_(), info_(new MetaInfo) {}
  void CreateInfo();

  using UptrDataSource = std::unique_ptr<DataSource<SparsePage>>;
  using FwdListType = std::forward_list<UptrDataSource>;

  class BatchSetIteratorImpl : public BatchIteratorImpl<SparsePage> {
   public:
    explicit BatchSetIteratorImpl(const FwdListType& sources)
        : sources_(sources), iter_(sources_.begin()) {}
    SparsePage& operator*() override {
      CHECK(!AtEnd());
      return dynamic_cast<SimpleCSRSource*>(iter_->get())->page_;
    }
    const SparsePage& operator*() const override {
      CHECK(!AtEnd());
      return dynamic_cast<const SimpleCSRSource*>(iter_->get())->page_;
    }
    void operator++() override { 
      ++iter_;
    }
    bool AtEnd() const override {
      return iter_ == sources_.end();
    }

   private:
    const FwdListType& sources_;
    FwdListType::const_iterator iter_;
  };

  static BatchedDMatrix* newMat_;
  static std::mutex batchMutex_;
  static std::condition_variable batchBuiltCondVar_;

  int nBatches_;
  int nSources_{0};
  std::forward_list<UptrDataSource> sources_;
  std::unique_ptr<MetaInfo> info_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
