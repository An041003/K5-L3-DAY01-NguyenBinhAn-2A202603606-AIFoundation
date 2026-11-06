# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Bản chuẩn bị nộp, tổng hợp từ notebook Colab và JSON do người học cung cấp. Chưa hoàn tất: thiếu ảnh tính tay và còn các điểm cần người học rà soát bên dưới. Notebook và JSON được giữ nguyên nội dung, output và thời gian đã ghi.

- **SEED:** 9163.
- **Kiểm tra lưu trong kết quả Colab:** 15/15 đạt.
- **Dự đoán đã ghi:** 10/10.
- **Thử thách cá nhân:** đề #2 (giảm vocab xuống 100 và tăng lên 5.000 từ).
- Họ tên và MSSV được thể hiện trong tên file theo hướng dẫn bảo vệ thông tin cá nhân.

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) | Đã có, 101 cell, có output Colab |
| [lab1_report.json](lab1_report.json) | Đã có; có `seed`, `checks`, `predictions`, `results` |
| `hand_calc.jpg` | **Chưa có**; người học xác nhận chưa có ảnh. Không thay thế bằng kết quả hash |

## 2. Kết quả chính

Số liệu dưới đây lấy trực tiếp từ `lab1_report.json`, không chạy lại thí nghiệm. Các mô hình 1–3 có F1 trên toàn bộ test; các hàng LLM chỉ được đánh giá trên SUB 40 tin. Chi phí là số notebook ước tính, không phải xác nhận hóa đơn của nhà cung cấp.

| Mô hình | F1 test | F1 SUB 40 | p95 ms/tin | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.8902 | 0.8889 | 0.099 | 0.000000 |
| 2. BoW + MLP (deep learning) | 0.9341 | 0.9189 | 0.316 | 0.000000 |
| 3a. Encoder (MiniLM) + kNN | 0.9293 | 1.0000 | 128.192 | 0.000000 |
| 3b. Encoder (MiniLM) + LogReg | 0.9529 | 0.9744 | 59.397 | 0.000000 |
| 3c. Encoder (MiniLM) + MLP head | 0.9534 | 0.9744 | 57.359 | 0.000000 |
| 4a. LLM nhỏ zero-shot | — | 1.0000 | 1426.136 | 0.008263 |
| 4b. LLM lớn zero-shot | — | 1.0000 | 1150.627 | 0.082633 |
| 4c. LLM nhỏ few-shot | — | 1.0000 | 1098.538 | 0.018663 |

## 3. Dự đoán và thực tế

Ba đối chiếu dưới đây lấy dự đoán từ JSON và so với bằng chứng đã lưu. Phần giải thích trải nghiệm cá nhân cần người học tự rà soát; đây không phải dự đoán mới được bổ sung sau thí nghiệm.

### `llm_models`

- Dự đoán được lưu lúc `06:47:44`: model lớn tốt hơn nhưng chênh không nhiều khoảng 10% đắt hơn khoảng 3 lần llm có thể không vượt vì embedding đang khá cao
- Đối chiếu bằng chứng: Cả hai LLM đạt F1 SUB = 1,0000; chi phí/1.000 tin của model lớn gấp 10 lần model nhỏ trong JSON.

### `llm_fewshot`

- Dự đoán được lưu lúc `08:21:54`: F1 giảm, chi phí tăng không đáng kể
- Đối chiếu bằng chứng: F1 SUB giữ nguyên 1,0000; chi phí tăng từ 0,008263 lên 0,018663 USD/1.000 tin (khoảng 2,26 lần).

### `hard_eval`

- Dự đoán được lưu lúc `09:09:32`: Dự đoán Accuracy của các mô hình sẽ giảm : BoW có thể đạt khoảng 60-75%, Embedding khoảng 70-85%, LLM khoảng 80-95%. BoW có thể gặp khó khăn nhất với nhóm ham chứa nhiều từ khoá spam LLM có thể gặp khó khăn với prompt injection và các tin nhắn lừa đảo được viết tự nhiên
- Đối chiếu bằng chứng: Output stress test cuối cùng đã gồm 60 case (50 gốc + 10 tự thêm), nên cần tách lại trước khi so sánh trực tiếp với dự đoán cho 50 case.


**Lưu ý về trình tự:** `llm_models` đang lưu dự đoán lúc 06:47:44, trong khi lịch sử có các lần chạy thí nghiệm từ 05:32:10. Có thể ô dự đoán đã được chạy lại; cần tự giải thích lịch sử này. Giữ nguyên mọi mốc thời gian, không sửa để tạo bằng chứng dự đoán trước thí nghiệm.

## 4. Phân tích 50 case khó

Output cuối của ô chạy stress test (execution count 124) được tạo sau ô thêm 10 `MY_CASES` (execution count 123). Vì vậy bảng còn mang nhãn “50 case” nhưng thực tế đánh giá **60 case: 50 gốc + 10 tự thêm**.

| Mô hình | F1 SUB thường | F1 stress test cuối (60 case) | Accuracy (60 case) |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0,889 | 0,326 | 0,517 |
| BoW + MLP | 0,919 | 0,391 | 0,533 |
| Embedding + kNN | 1,000 | 0,696 | 0,650 |
| Embedding + LogReg | 0,974 | 0,633 | 0,633 |
| Embedding + MLP head | 0,974 | 0,677 | 0,650 |
| LLM nhỏ zero-shot | 1,000 | 1,000 | 1,000 |
| LLM nhỏ + prompt phòng thủ | 1,000 | 1,000 | 1,000 |

Ô đọc lỗi vẫn lưu output của lần trước, với dòng `BoW + LogReg sai 25/50 case`. Trong output đó, case #11 (spam giọng tự nhiên) và #29 (spam thay chữ bằng số) đều bị BoW + LogReg dự đoán sai. Không dùng output cũ này làm danh sách lỗi của lần đánh giá 60 case.

**Cần rà soát:** tách kết quả 50 case gốc và 10 case bonus; cập nhật phần trả lời Câu hỏi 6 đang dùng số liệu cũ (ví dụ F1 0,286). Các case bonus hiện có `why_tricky` chung là “case của học viên”; cần bổ sung giải thích riêng nếu muốn đáp ứng rubric bonus. Không tự chạy lại API hoặc sửa output trong bước chuẩn bị này.

## 5. Thử thách cá nhân

Đề #2 theo `SEED % 6 = 1`: giảm vocab xuống 100 từ và tăng lên 5.000 từ; train lại, báo F1 test và giải thích xu hướng. Notebook đã có code và output cho đề này.

| Vocab | Precision | Recall | F1 test |
| ---: | ---: | ---: | ---: |
| 100 | 0,961 | 0,760 | 0,849 |
| 2.000 (baseline) | 1,000 | 0,802 | 0,890 |
| 5.000 | 1,000 | 0,802 | 0,890 |

Phần giải thích dưới đây được chép từ notebook của người học:

Khi giảm Vocabulary xuống 100 từ, mô hình đạt F1 = 0.849, Precision = 0.961 và Recall = 0.760. Nguyên nhân là từ điển quá nhỏ làm mất nhiều đặc trưng quan trọng, khiến mô hình bỏ sót nhiều tin Spam hơn.

Khi tăng Vocabulary lên 2.000 từ, F1 cải thiện lên 0.890 nhờ có thêm các từ mang thông tin phân loại hữu ích. Tuy nhiên, tiếp tục tăng lên 5.000 từ không làm F1 tăng thêm, vẫn đạt 0.890, với Precision = 1.000 và Recall = 0.802.

Kết quả cho thấy Vocabulary quá nhỏ có thể gây thiếu đặc trưng và làm giảm hiệu quả phân loại. Ngược lại, Vocabulary lớn hơn không đảm bảo mô hình tốt hơn vì các từ bổ sung có thể ít hữu ích hoặc xuất hiện quá hiếm.

## 6. Phương án cho sản phẩm thật

Phần dưới đây được chép từ mục phản ánh cuối bài trong notebook, chưa sửa nội dung của người học:

C1

Chỉ BoW + LR và BoW + MLP đạt SLA p95 dưới 50 ms. Chọn BoW + MLP vì có F1 cao hơn (0.934 so với 0.890). Tuy nhiên, chưa có mô hình nào vừa đạt SLA vừa đảm bảo F1 ≥ 0.95 trên toàn bộ tập test.

Nếu xử lý 300 triệu SMS/tháng bằng LLM, chi phí ước tính khoảng 975 USD với GPT-6 Luna hoặc 9.900 USD với GPT-6.1 Terra, dựa trên kết quả batching. LLM có F1 cao nhưng độ trễ và chi phí lớn.

C2

Đề xuất kết hợp BoW + MLP và LLM. MLP xử lý phần lớn SMS: p < 0.3 dự đoán Ham, p > 0.7 dự đoán Spam, còn p từ 0.3 đến 0.7 chuyển sang LLM để phân tích ngữ nghĩa.

Hệ thống này giúp giảm chi phí, tăng tốc độ và xử lý tốt hơn các tin nhắn khó. Tuy nhiên, cần điều chỉnh ngưỡng và giới hạn tỷ lệ gọi LLM để đảm bảo SLA 50 ms.

C3

Sai khi learning rate = 50 vẫn hội tụ và MLP bỏ ReLU vẫn đạt F1 = 0.934. Ngoài ra, LLM nhỏ đạt F1 = 1.000, ngang model lớn trên tập thử nghiệm.

Qua đó, em hiểu rằng mô hình phức tạp hơn chưa chắc tốt hơn. Việc lựa chọn AI phải cân bằng giữa chất lượng, tốc độ, chi phí và khả năng tổng quát hóa.

**Điểm cần đối chiếu trước khi nộp:** với 300 triệu SMS/tháng và `usd_per_1k` trong JSON hiện tại, phép nhân cho ra khoảng **2,479.00 USD/tháng** cho LLM nhỏ zero-shot và **24,790.00 USD/tháng** cho LLM lớn zero-shot. Hai số này chưa khớp 975/9.900 USD trong đoạn phản ánh; cần làm rõ giả định batching hoặc sửa phần giải thích bằng số liệu được kiểm chứng. Không thay đổi số đo gốc.

## 7. AI log và điều đã học

Log dưới đây ghi phần hỗ trợ có bằng chứng trong phiên làm việc này. Người học cần bổ sung trung thực các lần dùng AI khác trên Colab nếu có; không suy đoán hay điền thay lịch sử chưa được cung cấp.

| Phần | Yêu cầu với AI | Hỗ trợ đã thực hiện | Kiểm chứng / phần người học cần bổ sung |
| --- | --- | --- | --- |
| GitHub và môi trường | Đẩy dự án lên GitHub, kiểm tra và cài môi trường chạy trên máy | Tạo repo; cài `.venv`, Jupyter và thư viện; kiểm tra dữ liệu SMS, MiniLM | Kernel, dữ liệu và embedding đã được kiểm tra ở phiên chuẩn bị môi trường |
| Hiển thị notebook | Hỏi vì sao notebook không hiện từng cell | Kiểm tra định dạng; đặt `.ipynb` mở bằng Notebook Editor | Cấu hình tại `.vscode/settings.json` |
| Chuẩn bị bài nộp | Đổi tên dự án, đồng bộ GitHub/máy và sắp xếp kết quả Colab | Di chuyển/đổi tên hai file kết quả; tổng hợp bảng và phần trả lời có sẵn; lập danh sách điểm chưa khớp | Không viết thêm code TODO, không chạy lại API, không sửa số đo hoặc thời gian trong hai file kết quả |
| Các lần dùng AI khác | Chưa được cung cấp | Không tự điền | Người học tự bổ sung nếu có |

Điều đã học, trích từ phản ánh của người học: “mô hình phức tạp hơn chưa chắc tốt hơn. Việc lựa chọn AI phải cân bằng giữa chất lượng, tốc độ, chi phí và khả năng tổng quát hóa.” Người học có thể bổ sung thêm bằng trải nghiệm của mình.
