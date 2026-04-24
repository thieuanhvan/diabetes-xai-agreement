THƯ MỤC: experiments/analysis
README – Phân tích dữ liệu và đánh giá mô hình

Mục đích
Thư mục này chứa các script phân tích (analysis scripts) phục vụ nghiên cứu và luận văn.
Các file có tiền tố `run_` là các script dùng để chạy các thí nghiệm phân tích dữ liệu, đánh giá mô hình, và tạo các hình ảnh/bảng kết quả phục vụ báo cáo, slides và bài báo.

Tất cả các script trong thư mục này **không huấn luyện mô hình chính của pipeline**, mà chỉ sử dụng dữ liệu hoặc kết quả từ pipeline để thực hiện các phân tích bổ sung.

---

QUY TẮC OUTPUT

Tất cả các script `run_*` trong thư mục này chỉ tạo output trong:

outputs/analysis/

Các file được sinh ra có thể bao gồm:

* hình ảnh (.png)
* bảng kết quả (.csv)
* báo cáo tóm tắt (.txt)

Thư mục outputs chính của project có thể có nhiều thư mục con, nhưng các script trong `analysis` chỉ ghi dữ liệu vào:

outputs/analysis/

Điều này giúp tách biệt:

* outputs/pipeline → kết quả huấn luyện mô hình
* outputs/analysis → kết quả phân tích dữ liệu và mô hình

---

MÔ TẢ CÁC SCRIPT RUN_

1. run_eda_all_datasets.py

Mục đích
Thực hiện Exploratory Data Analysis (EDA) cho tất cả các dataset đã đăng ký trong dataset registry.

Các phân tích bao gồm:

* phân bố class
* phân bố của từng feature
* correlation heatmap
* feature importance cơ bản (Random Forest)

Output sinh ra:

outputs/analysis/eda/

Ví dụ file output:

class_distribution_<dataset>.png
feature_distributions_<dataset>.png
correlation_heatmap_<dataset>.png
feature_importance_rf_<dataset>.png

---

2. run_error_feature_analysis.py

Mục đích
Phân tích các lỗi dự đoán của mô hình, đặc biệt là:

* False Positive (FP)
* False Negative (FN)

Script này so sánh các đặc trưng của các mẫu bị dự đoán sai để tìm ra:

* feature nào làm mô hình dễ sai
* pattern của lỗi dự đoán

Output sinh ra:

outputs/analysis/error_analysis/

Ví dụ file output:

error_feature_analysis_<dataset>.csv
top_features_false_negative_<dataset>.csv

---

3. run_explainability_analysis.py

Mục đích
Thực hiện phân tích Explainable AI bằng SHAP để giải thích quyết định của mô hình.

Các phân tích bao gồm:

* SHAP summary plot
* SHAP feature importance
* ranking các feature quan trọng

Output sinh ra:

outputs/analysis/explainability/

Ví dụ file output:

shap_summary_<dataset>.png
shap_feature_importance_<dataset>.csv

---

4. run_feature_agreement_analysis.py

Mục đích
So sánh độ quan trọng của feature giữa nhiều mô hình khác nhau.

Ví dụ:

* Random Forest
* XGBoost
* SHAP importance
* các phương pháp khác

Mục tiêu là kiểm tra xem các mô hình có đồng ý với nhau về feature quan trọng hay không.

Output sinh ra:

outputs/analysis/feature_agreement/

Ví dụ file output:

feature_importance_comparison_<dataset>.csv

---

5. run_model_stability_analysis.py

Mục đích
Kiểm tra độ ổn định của mô hình khi thay đổi random seed hoặc khi chạy nhiều lần training.

Script này giúp xác định:

* mô hình có ổn định hay không
* kết quả có phụ thuộc quá nhiều vào random seed hay không

Output sinh ra:

outputs/analysis/model_stability/

Ví dụ file output:

model_stability_results.csv

---

6. run_cross_dataset_generalization.py

Mục đích
Đánh giá khả năng tổng quát hóa của mô hình giữa các dataset khác nhau.

Ví dụ:

train trên dataset A
test trên dataset B

Phân tích này giúp kiểm tra mô hình có thể áp dụng cho dữ liệu từ nguồn khác hay không.

Output sinh ra:

outputs/analysis/generalization/

Ví dụ file output:

cross_dataset_results.csv

---

7. run_ablation_study.py

Mục đích
Thực hiện ablation study để đánh giá mức độ quan trọng của từng nhóm feature.

Phương pháp:

* loại bỏ từng nhóm feature
* huấn luyện lại mô hình
* so sánh hiệu năng

Output sinh ra:

outputs/analysis/ablation/

Ví dụ file output:

ablation_results.csv

---

8. run_smote_preprocessing.py

Mục đích
Áp dụng kỹ thuật SMOTE để xử lý mất cân bằng dữ liệu.

Script này tạo ra dataset mới sau khi oversampling.

Output sinh ra:

outputs/analysis/smote/

Ví dụ file output:

dataset_smote_<dataset>.csv

---

9. run_smote_comparison.py

Mục đích
So sánh hiệu năng của mô hình:

* trước khi dùng SMOTE
* sau khi dùng SMOTE

Output sinh ra:

outputs/analysis/smote_comparison/

Ví dụ file output:

smote_comparison_results.csv

---

10. run_statistical_tests.py

Mục đích
Thực hiện các kiểm định thống kê trên dataset.

Ví dụ:

* t-test
* chi-square test
* kiểm tra sự khác biệt giữa các nhóm

Output sinh ra:

outputs/analysis/statistical_tests/

Ví dụ file output:

statistical_test_results.csv

---

11. run_statistical_tests_basic.py

Mục đích
Phiên bản đơn giản hơn của statistical tests để kiểm tra nhanh các feature.

Output sinh ra:

outputs/analysis/statistical_tests_basic/

---

CÁC FILE KHÔNG PHẢI RUN SCRIPT

Ngoài các file run_ còn có các module hỗ trợ:

dataset_profiler.py
dataset_visualizer.py
dataset_comparator.py
shap_explainer.py
io_utils.py

Các file này chỉ là thư viện hỗ trợ cho các script run_ và không chạy trực tiếp.

---

KẾT LUẬN

Các script trong thư mục này phục vụ cho các phân tích nghiên cứu như:

* Exploratory Data Analysis (EDA)
* Explainable AI
* Error analysis
* Ablation study
* Model stability
* Cross-dataset evaluation
* Statistical validation

Tất cả output của các script này được lưu trong:

outputs/analysis/

để tách biệt với các output của pipeline huấn luyện mô hình.
