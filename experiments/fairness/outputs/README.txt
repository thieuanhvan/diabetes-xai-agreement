FAIRNESS THEO NHÓM TUỔI (AGE GROUPS)
==================================

Thư mục này chứa kết quả đánh giá fairness cho mô hình Gradient Boosting
được đề xuất, dựa trên so sánh hiệu năng giữa các nhóm tuổi.

Các nhóm tuổi:
- <40
- 40–60
- >60

Các chỉ số được đánh giá:
- Recall (TPR)
- False Positive Rate (FPR)
- ROC-AUC

Các tập tin
-----------
1. fairness_age_metrics.csv
   - Bảng số liệu chi tiết theo nhóm tuổi.

2. fairness_recall_by_age.png
   - Biểu đồ so sánh Recall (TPR) theo nhóm tuổi.

3. fairness_fpr_by_age.png
   - Biểu đồ so sánh FPR theo nhóm tuổi.

4. fairness_roc_auc_by_age.png
   - Biểu đồ so sánh ROC-AUC theo nhóm tuổi.

Thông tin thực thi
------------------
- Thời điểm chạy : 20260206_215804
- Thời gian chạy : 71.32 giây

Ý nghĩa
-------
Các biểu đồ fairness giúp trực quan hóa sự khác biệt về hành vi của mô hình
giữa các nhóm tuổi, làm cơ sở cho các phân tích Fairness + Explainable AI
trong các bước tiếp theo.