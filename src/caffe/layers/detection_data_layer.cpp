#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/detection_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
namespace caffe {

template <typename Dtype>
DetectionDataLayer<Dtype>::DetectionDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
DetectionDataLayer<Dtype>::~DetectionDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  BBoxDatum bbox_datum;
  bbox_datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  const Datum& datum = bbox_datum.datum();
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // Images in each batch may contain different number of bboxes, while Blob
  // assumes identical spatial size across channels within a batch. Thus,
  // use a single channel to put together all bboxes in a batch.
  if (this->output_labels_) {
    // label blob shape [#bboxes, 6]
    vector<int> label_shape(2, 1);
    // #bboxes in a batch, will reshape later
    label_shape[0] = 1;
    // [item_id, label, cx, cy, w, h]
    label_shape[1] = 6;
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
  }
}

// In case of multi-gpu training, multiple solvers will open db in their
// own thread, and a solver reads every solver_count datums in db, and
// use this function to skip others.
template <typename Dtype>
bool DetectionDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void DetectionDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void DetectionDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  BBoxDatum bbox_datum;
  vector<vector<BBox> > bboxes(batch_size, vector<BBox>());
  int num_bboxes = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    bbox_datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      const Datum& datum = bbox_datum.datum();
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }

    // Apply data and label transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(bbox_datum, &(this->transformed_data_));

    for (int i = 0; i < bbox_datum.bbox_size(); i++) {
      bboxes[item_id].push_back(bbox_datum.bbox(i));
    }
    num_bboxes += bbox_datum.bbox_size();
    trans_time += timer.MicroSeconds();
    Next();
  }

  // Copy label.
  if (this->output_labels_) {
    vector<int> label_shape(2);
    label_shape[0] = num_bboxes;
    label_shape[1] = 6;
    batch->label_.Reshape(label_shape);

    int offset = 0;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      for (int i = 0; i < bboxes[item_id].size(); i++) {
        Dtype* top_label = batch->label_.mutable_cpu_data();
        const BBox& bbox = bboxes[item_id][i];
        top_label[offset++] = item_id;
        top_label[offset++] = bbox.label();
        top_label[offset++] = bbox.center_x();
        top_label[offset++] = bbox.center_y();
        top_label[offset++] = bbox.width();
        top_label[offset++] = bbox.height();
      }
    }
  }

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DetectionDataLayer);
REGISTER_LAYER_CLASS(DetectionData);

}  // namespace caffe
