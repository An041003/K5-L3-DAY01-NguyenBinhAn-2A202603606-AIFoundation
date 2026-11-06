# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 9163.
- **Kết quả kiểm tra:** 15/15 đạt.
- **Dự đoán đã ghi:** 10/10.
- **Thử thách cá nhân:** đề #2 — thay đổi kích thước vocabulary.

## 1. Bằng chứng

Bài lab được thực hiện trên Colab. Các tệp minh chứng gồm:

- [day01_lab.ipynb](day01_lab.ipynb): code, câu trả lời, biểu đồ và output của các thí nghiệm.
- [lab1_report.json](lab1_report.json): SEED, kết quả kiểm tra, các dự đoán và bảng số liệu tổng hợp.
- [hand_calc.jpg](hand_calc.jpg): ảnh các bước tính tay cho năm bài tính trong lab.

## 2. Kết quả chính

Các phương pháp được so sánh trên cùng cách chia dữ liệu. Với ML, DL và Embedding, kết quả được đánh giá trên toàn bộ tập test và tập SUB 40 tin. Với LLM, kết quả được đánh giá trên tập SUB để giới hạn số lần gọi API.

| Mô hình | F1 test | F1 SUB 40 | p95 ms/tin | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.8902 | 0.8889 | 0.099 | 0.000000 |
| 2. BoW + MLP | 0.9341 | 0.9189 | 0.316 | 0.000000 |
| 3a. Encoder MiniLM + kNN | 0.9293 | 1.0000 | 128.192 | 0.000000 |
| 3b. Encoder MiniLM + LogReg | 0.9529 | 0.9744 | 59.397 | 0.000000 |
| 3c. Encoder MiniLM + MLP head | 0.9534 | 0.9744 | 57.359 | 0.000000 |
| 4a. LLM nhỏ zero-shot | — | 1.0000 | 1426.136 | 0.008263 |
| 4b. LLM lớn zero-shot | — | 1.0000 | 1150.627 | 0.082633 |
| 4c. LLM nhỏ few-shot | — | 1.0000 | 1098.538 | 0.018663 |

Embedding + MLP head có F1 cao nhất trên toàn bộ tập test, đạt khoảng 0.9534. Tuy nhiên, BoW + MLP nhanh hơn nhiều và vẫn đạt F1 khoảng 0.9341. Trên SUB 40 tin, cả ba cấu hình LLM đều đạt F1 = 1.000, nhưng độ trễ cao hơn các phương pháp còn lại.

## 3. Dự đoán và thực tế

### Learning rate lớn

Dự đoán ban đầu: `lr = 50` có thể làm loss dao động mạnh hoặc tăng đột ngột vì bước cập nhật quá lớn. Kết quả khớp một phần: loss tăng vọt ở giai đoạn đầu nhưng sau đó giảm nhanh và đạt khoảng 0.004. Trong khi đó, `lr = 0.001` học chậm, loss cuối khoảng 0.627; `lr = 0.5` giảm ổn định xuống khoảng 0.081.

Thí nghiệm cho thấy learning rate lớn có thể gây mất ổn định ban đầu nhưng không có nghĩa mô hình chắc chắn không hội tụ.

### Model LLM lớn và nhỏ

Dự đoán ban đầu: model lớn có F1 cao hơn khoảng 10% và đắt hơn khoảng 3 lần. Thực tế, cả GPT-6 Luna và GPT-6.1 Terra đều đạt F1 = 1.000 trên SUB 40 tin. Chi phí ước tính của model lớn lại gấp 10 lần model nhỏ, từ khoảng 0.008263 lên 0.082633 USD/1.000 tin.

Điểm bất ngờ là model nhỏ đã phân loại đúng toàn bộ tập SUB, nên model lớn không tạo thêm khác biệt về F1 trên tập này.

### Few-shot

Dự đoán ban đầu: F1 có thể giảm và chi phí tăng không đáng kể. Thực tế, F1 vẫn giữ nguyên ở mức 1.000, còn chi phí tăng từ khoảng 0.008263 lên 0.018663 USD/1.000 tin, tức khoảng 2.26 lần.

Việc thêm ví dụ làm prompt dài hơn và tăng số token input. Với kết quả của thí nghiệm này, few-shot chưa đem lại lợi ích về F1 so với zero-shot.

## 4. Phân tích các case khó

Các mô hình được thử nghiệm trên bộ 50 case khó và 10 case tự xây dựng. Kết quả của lần chạy trên tổng cộng 60 case như sau:

| Mô hình | F1 SUB thường | F1 trên 60 case khó | Accuracy trên 60 case |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.889 | 0.326 | 0.517 |
| BoW + MLP | 0.919 | 0.391 | 0.533 |
| Embedding + kNN | 1.000 | 0.696 | 0.650 |
| Embedding + LogReg | 0.974 | 0.633 | 0.633 |
| Embedding + MLP head | 0.974 | 0.677 | 0.650 |
| LLM nhỏ zero-shot | 1.000 | 1.000 | 1.000 |
| LLM nhỏ + prompt phòng thủ | 1.000 | 1.000 | 1.000 |

BoW + LogReg giảm F1 nhiều nhất. Nguyên nhân là mô hình dựa nhiều vào các từ khóa đã gặp trong train, còn bộ case khó có nhiều tin quảng cáo được viết tự nhiên hoặc cố tình thay đổi từ khóa.

Trong các tin BoW + LogReg dự đoán sai, hai trường hợp đáng chú ý là:

- **Case #11:** tin nhắn mở đầu như lời trò chuyện từ người quen ở phòng gym nhưng thực chất đang quảng cáo gói thành viên. Cách viết tự nhiên khiến BoW khó nhận diện ý định quảng cáo.
- **Case #29:** tin nhắn dùng các từ như `C0ngr4tulat10ns`, `w0n` và `g1ft c4rd`. Việc thay chữ bằng số làm các token khác với những từ quen thuộc trong vocabulary.

Embedding giữ được F1 tốt hơn BoW trên bộ case khó vì biểu diễn được ngữ nghĩa của văn bản. Tuy nhiên, các mô hình này vẫn có thể nhầm giữa tin lừa đảo và tin đang nói về lừa đảo.

Cả LLM zero-shot và LLM có prompt phòng thủ đều đạt Accuracy = 100% trong lần kiểm tra này. Prompt phòng thủ không làm tăng điểm vì cấu hình zero-shot đã phân loại đúng toàn bộ các case.

## 5. Thử thách cá nhân

Với SEED = 9163, đề #2 yêu cầu: giảm vocabulary xuống 100 từ và tăng lên 5.000 từ, train lại rồi so sánh F1 trên tập test.

| Vocabulary | Precision | Recall | F1 test |
| ---: | ---: | ---: | ---: |
| 100 | 0.961 | 0.760 | 0.849 |
| 2.000 | 1.000 | 0.802 | 0.890 |
| 5.000 | 1.000 | 0.802 | 0.890 |

Khi giảm vocabulary xuống 100 từ, mô hình đạt F1 = 0.849, Precision = 0.961 và Recall = 0.760. Từ điển quá nhỏ làm mất nhiều đặc trưng quan trọng, khiến mô hình bỏ sót nhiều tin spam hơn.

Khi tăng vocabulary lên 2.000 từ, F1 cải thiện lên 0.890 nhờ có thêm các từ mang thông tin phân loại hữu ích. Tuy nhiên, tăng tiếp lên 5.000 từ không làm F1 tăng thêm, Precision và Recall cũng giữ nguyên.

Kết quả cho thấy vocabulary quá nhỏ có thể làm thiếu đặc trưng, còn vocabulary lớn hơn chưa chắc giúp mô hình tốt hơn vì các từ bổ sung có thể ít hữu ích hoặc xuất hiện quá hiếm.

## 6. Phương án cho sản phẩm thật

Với yêu cầu xử lý 10 triệu SMS/ngày và p95 dưới 50 ms/tin, lựa chọn phù hợp là BoW + MLP. Mô hình này đạt F1 khoảng 0.934 và p95 khoảng 0.316 ms/tin, cao hơn về F1 so với BoW + LogReg nhưng vẫn đáp ứng yêu cầu độ trễ.

Các phương pháp dùng embedding có F1 tốt hơn trên toàn bộ tập test, nhưng p95 khoảng 57–128 ms/tin nên vượt mức 50 ms trong kết quả thực nghiệm. LLM đạt F1 cao trên SUB nhưng có độ trễ lớn hơn nhiều và phát sinh chi phí API.

Đề xuất kết hợp BoW + MLP với LLM theo cách định tuyến:

- `p < 0.3`: MLP phân loại ham.
- `p > 0.7`: MLP phân loại spam.
- `p` nằm trong khoảng `[0.3, 0.7]`: chuyển sang LLM để xử lý.

Cách này cho phép MLP xử lý phần lớn tin nhắn với tốc độ cao, còn LLM tập trung vào những tin khó. Khi triển khai, cần điều chỉnh ngưỡng và giới hạn tỷ lệ chuyển sang LLM để kiểm soát độ trễ và chi phí.

## 7. AI log và điều đã học

AI được sử dụng để hỗ trợ thiết lập môi trường, thao tác GitHub và trình bày báo cáo từ các kết quả đã chạy trên Colab.

| Phần | Yêu cầu với AI | AI hỗ trợ gì | Kết quả |
| --- | --- | --- | --- |
| GitHub và môi trường | Đẩy dự án lên GitHub, kiểm tra khả năng chạy trên máy và cài thư viện | Thiết lập repo, `.venv`, Jupyter và các thư viện cần thiết | Môi trường được kiểm tra bằng dữ liệu SMS và model MiniLM |
| Hiển thị notebook | Vì sao file notebook không hiện thành từng cell | Kiểm tra định dạng và cấu hình Notebook Editor trong VS Code | Có cấu hình mở file `.ipynb` bằng Notebook Editor |
| Chuẩn bị bài nộp | Đổi tên dự án và sắp xếp các file kết quả Colab theo hướng dẫn | Đổi tên, di chuyển file, tổng hợp số liệu và câu trả lời có sẵn | Notebook và JSON giữ nguyên nội dung cùng output |
| Trình bày báo cáo | Viết lại báo cáo theo văn phong báo cáo cá nhân | Chỉnh văn phong trực tiếp, lược bỏ đại từ xưng hô và sắp xếp lại nội dung | Báo cáo sử dụng số liệu thực nghiệm và phần giải thích trong notebook |

Điều bất ngờ nhất là `lr = 50` vẫn hội tụ sau giai đoạn mất ổn định, và LLM nhỏ đạt F1 ngang model lớn trên tập SUB. Qua bài lab, có thể thấy mô hình phức tạp hơn chưa chắc phù hợp hơn. Việc lựa chọn phương pháp cần dựa vào cả chất lượng, tốc độ, chi phí và khả năng xử lý những trường hợp khó.
