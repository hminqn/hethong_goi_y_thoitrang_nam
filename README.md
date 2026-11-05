# FORM AI — Hệ thống gợi ý phối đồ thời trang nam

Tác giả: [hminqn](https://github.com/hminqn).

Mã nguồn hiện tại được viết lại trong **[fashion-ai/](fashion-ai/)**. Hệ thống xử lý ảnh và suy luận trên máy, không sử dụng dịch vụ API bên ngoài khi chạy.

## Luồng hoạt động

Ảnh người dùng → nhận diện và phân đoạn → chọn món muốn giữ → trích xuất đặc trưng → tìm món bổ sung theo luật → chấm điểm tương thích → chọn tối đa 5 bộ áo–quần–giày.

## Dataset và thuật toán

| Thành phần | Dataset | Thuật toán |
| --- | --- | --- |
| Nhận diện, phân đoạn trang phục | Fashionpedia | Mask R-CNN, ResNet-50 + FPN |
| Học đặc trưng, tìm món tương tự | Polyvore Outfits ECCV 2018, disjoint | ResNet-50 + Triplet Loss, cosine similarity |
| Khai phá quy luật phối đồ | Polyvore train | FP-Growth, so sánh Apriori |
| Phân loại độ tương thích bộ đồ | Polyvore disjoint | Random Forest, so sánh Logistic Regression |
| Kết hợp và xếp hạng gợi ý | Polyvore valid/test | Điểm có trọng số và MMR |

Xem [hướng dẫn cài đặt, huấn luyện, đánh giá và phân công](fashion-ai/README.md).

## Chạy website

Cần Python 3.12 và các artifact đã được tạo bằng các bước huấn luyện trong tài liệu.

```powershell
cd fashion-ai
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m fashion_ai.app
```

Mở **http://127.0.0.1:5000**. Website hiển thị trạng thái từng mô hình. Khi Fashionpedia chưa có checkpoint, có thể tự khoanh vùng món trong ảnh hoặc chọn món từ kho đã chuẩn bị để kiểm tra phần gợi ý.

## Huấn luyện Fashionpedia đầy đủ

Trong thư mục fashion-ai, chạy:

```powershell
.\train-fashionpedia.ps1
```

Script tải toàn bộ ảnh/nhãn từ nguồn chính thức, kiểm tra dữ liệu, rồi huấn luyện 10 epoch trên 45.623 ảnh train và đánh giá trên 1.158 ảnh validation. Cấu hình CPU: batch 1, cạnh ngắn 512, cạnh dài tối đa 768. Lưu checkpoint mỗi 100 batch và hỗ trợ tiếp tục khi bị ngắt. Website tự cập nhật tiến độ. Quy trình đã được khởi động trên máy phát triển; chưa công bố checkpoint Fashionpedia hoàn chỉnh.

## Kết quả ban đầu

Đã huấn luyện projection của encoder 2 epoch, trích xuất embedding, khai phá 206 luật, huấn luyện RF/LR và chọn cấu hình MMR. RF đạt ROC-AUC khoảng 0,535 trên test âm tính tổng hợp; Hit@5 xếp hạng khoảng 0,274 theo phép thử tự xây dựng. Kết quả còn thấp, cần cải thiện trước khi kết luận chất lượng. Các giới hạn và giao thức đánh giá được ghi trong [tài liệu mô hình](fashion-ai/README.md).

12 kiểm thử đã đạt, gồm tiếp tục huấn luyện đúng batch, tách dữ liệu, luật, MMR và luồng tải ảnh → khoanh vùng → gợi ý với checkpoint thật trên máy phát triển.

```powershell
cd fashion-ai
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Phạm vi repository

Repository chứa mã nguồn và hướng dẫn tái lập. Dataset, ảnh sản phẩm, môi trường .venv, cache và artifact huấn luyện nằm trong .gitignore; cần tải dữ liệu và chạy các bước chuẩn bị/huấn luyện khi cài trên máy mới. Các thư mục web/, scripts/, tests/ và data/ ở gốc là bản triển khai JavaScript trước đây; ứng dụng mới dùng fashion-ai/.

Nguồn dữ liệu: [Fashionpedia](https://github.com/cvdfoundation/fashionpedia), [Polyvore mirror Stylique](https://huggingface.co/datasets/Stylique/Polyvore). Giữ các điều kiện sử dụng và trích dẫn đi kèm dữ liệu gốc.
