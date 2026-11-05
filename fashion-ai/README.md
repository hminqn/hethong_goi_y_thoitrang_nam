# FORM AI — Gợi ý phối đồ nam từ ảnh

Mã mới viết trong `fashion-ai/`, triển khai năm mô-đun theo phân công. Giao diện phục vụ tại localhost, suy luận bằng mô hình trong máy. Dataset Polyvore đã tải ở thư mục cha được dùng làm đầu vào.

## Cài đặt và chạy

Dùng Python 3.12. Môi trường `.venv` trên máy hiện tại đã được cài thư viện trong `requirements.txt`. Trên máy khác, tạo môi trường và cài thư viện:

```powershell
cd fashion-ai
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m fashion_ai.app
```

Mở http://127.0.0.1:5000. Mặc định chỉ phục vụ trên máy đang chạy, debug tắt. Ảnh được gửi từ trình duyệt tới tiến trình localhost, các vùng ảnh được giữ tạm trong bộ nhớ tối đa 8 phiên. Khởi động lại ứng dụng để nạp lại checkpoint vừa huấn luyện.

## Chuẩn bị Polyvore

```powershell
.venv\Scripts\python.exe -m fashion_ai.train prepare
```

Mặc định đọc `../datasets/Stylique-Polyvore`. Có thể truyền `--source` với đường dẫn khác. Lọc nam dựa trên đường dẫn danh mục, Gender: Male hoặc category ID chỉ có nhãn male. Ưu tiên loại thông tin Female. Danh mục mơ hồ được loại. Mỗi món ghi bằng chứng lọc.

Xử lý theo thứ tự train, valid, test. Loại toàn bộ bộ ở split sau nếu có món đã xuất hiện ở split trước, kể cả khi bản mirror mang tên disjoint. Loại tập hợp món trùng trong mỗi split. Trích ảnh theo item ID từ images.zip, ghi manifest và số bộ bị loại. Mỗi lần chuẩn bị lại cần huấn luyện lại toàn bộ artifact để tránh trộn các phiên bản dữ liệu.

## Người 1 — Mask R-CNN ResNet-50 + FPN

Tải Fashionpedia từ nguồn chính thức, gồm ảnh train/val và `instances_attributes_train2020.json`, `instances_attributes_val2020.json`.

Nguồn https://github.com/cvdfoundation/fashionpedia

Quy trình đầy đủ trên máy này (tải tiếp dữ liệu, kiểm tra ZIP/ảnh/nhãn, sau đó huấn luyện 10 epoch và đánh giá toàn bộ validation):

```powershell
.\train-fashionpedia.ps1
```

Chạy nền, ghi tiến độ tại `artifacts/detector-status.json`, log tại `artifacts/fashionpedia-training.log` và lỗi tại `artifacts/fashionpedia-training-errors.log`. Website tự cập nhật tiến độ mỗi 15 giây. Nếu máy khởi động lại, chạy lại script để tiếp tục. Giữ máy bật trong quá trình chạy. Không tự cấu hình chạy cùng Windows.

Cấu hình CPU: batch 1, cạnh ngắn 512, cạnh dài tối đa 768, toàn bộ các lớp Fashionpedia, toàn bộ ảnh train ở mỗi epoch, không lấy mẫu nhỏ. SGD, LR 0,005, momentum 0,9, weight decay 0,0005; giảm LR mỗi 3 epoch. Lưu model/optimizer/scheduler và vị trí batch mỗi 100 batch vào `detector-latest.pt`. Khi tiếp tục, kiểm tra hash nhãn và cấu hình, giữ thứ tự xáo trộn theo epoch. Mô hình khởi tạo từ trọng số COCO chính thức; ba block cuối của backbone được fine-tune. Chỉ công bố `detector.pt` sau khi hoàn tất huấn luyện và đánh giá.

Nếu cần dừng an toàn: tạo tệp rỗng `artifacts/STOP_DETECTOR`. Xử lý xong batch hiện tại, chương trình lưu checkpoint và dừng. Xóa cờ này trước khi chạy lại. Nếu ngắt nguồn đột ngột, tiếp tục từ checkpoint đã ghi gần nhất.

Lệnh tùy chỉnh trực tiếp:

```powershell
.venv\Scripts\python.exe -m fashion_ai.train detector --train-annotations PATH\instances_attributes_train2020.json --train-images PATH\train --valid-annotations PATH\instances_attributes_val2020.json --valid-images PATH\val --epochs 10 --batch-size 2
```

Khởi tạo từ checkpoint COCO chính thức, thay cả head box/mask theo ánh xạ category Fashionpedia. Dataset đọc COCO polygons/RLE, loại crowd và mask/box rỗng. Đánh giá bbox AP và mask AP bằng COCO evaluator qua torchmetrics. Giao diện chỉ cho giữ áo, quần hoặc giày trong các nhãn đã ánh xạ.

Bàn giao `artifacts/detector.pt`, `detector-report.json`, `vision.py` và phần huấn luyện trong `train.py`. Checkpoint COCO gốc chưa được coi là mô hình Fashionpedia. Khi detector chưa được huấn luyện, giao diện hiển thị trạng thái chờ; có thể thử các phần gợi ý bằng ảnh tự khoanh vùng hoặc món trong kho.

## Người 2 — ResNet-50 + Triplet Loss

```powershell
.venv\Scripts\python.exe -m fashion_ai.train encoder --epochs 5 --batch-size 8 --dimension 128
```

Backbone ImageNet ResNet-50 được đóng băng để huấn luyện trên CPU khả thi; projection 2048 → 128 được học bằng Triplet Margin Loss, margin 0,2. Positive là món khác cùng category ID. Negative khác category ID và cùng vai trò. Nếu category chỉ có một món, positive là augmentation của anchor, có thống kê trong báo cáo. Đây là supervision theo nhãn danh mục.

Chỉ lấy triplet từ train. Sau huấn luyện, trích vector L2 cho các ảnh ở cả ba tập bằng cùng encoder đã học. Retrieval report đánh giá cùng category ID và cùng vai trò trên test, loại chính ảnh query khỏi gallery, báo Recall@1/5 và MRR. Chưa đánh giá độ giống hình ảnh bằng nhãn chuyên gia độc lập.

Bàn giao `encoder.pt`, `embeddings.npz`, `encoder-report.json`, `retrieval-report.json`.

## Người 3 — FP-Growth và Apriori

```powershell
.venv\Scripts\python.exe -m fashion_ai.train rules --support 0.05 --confidence 0.5
```

Mỗi transaction là tập thuộc tính của một bộ train có từ hai món. Thuộc tính gồm vai trò, category ID và các từ chỉ màu xuất hiện rõ trong tên sản phẩm. Màu từ tên có thể khác màu thực trên ảnh. Chỉ giữ luật giữa các vai trò khác nhau, lift > 1. Cùng dữ liệu và ngưỡng được dùng để đối chiếu thời gian FP-Growth/Apriori.

Bàn giao `rules.json`, `rules-report.json`, mã `mine_rules`/`rule_score` trong `recommendation.py`.

## Người 4 — Random Forest và Logistic Regression

```powershell
.venv\Scripts\python.exe -m fashion_ai.train compatibility
```

Vector bộ là mean/std các embedding, thống kê cosine từng cặp và tỉ lệ các vai trò. Biểu diễn cố định kích thước và không phụ thuộc thứ tự món. Mỗi split tạo âm tính bằng thay một món bằng món khác cùng vai trò trong chính split, loại các bộ dương đã biết và âm tính trùng.

Đây là bài tương thích với âm tính tổng hợp cho tập nam. Các bộ dương là phần món nam được giữ từ bộ gốc nên có thể có nhiễu. Random Forest được triển khai theo phương án chốt; Logistic Regression dùng StandardScaler học trên train và được báo cạnh RF. Ngưỡng phân loại chọn bằng F1 trên validation, thử 0,05 đến 0,95; ngưỡng được lưu trong `compatibility-thresholds.json`. Validation thiếu hai lớp thì dùng 0,5. Báo ROC-AUC, F1, Precision, Recall và số mẫu mỗi split. Có thêm phép đánh giá trên các mẫu test tương thích chính thức mà toàn bộ món đều thuộc tập nam đã giữ; các mẫu trộn hoặc thiếu món bị loại hoàn toàn.

Bàn giao `RandomForest.joblib`, `LogisticRegression.joblib`, `compatibility-report.json`.

## Người 5 — Điểm trọng số và MMR

```powershell
.venv\Scripts\python.exe -m fashion_ai.train ranking
```

Trên validation, thử 3 bộ trọng số RF/luật/cosine và 3 lambda MMR. Chọn theo Hit@5, MRR@5 rồi độ đa dạng. Sau đó đánh giá một lần trên test và ghi ablation chỉ RF, chỉ luật, chỉ cosine. Mỗi câu hỏi giữ lại các món ngữ cảnh, giấu một món rồi thêm tối đa 19 âm tính cùng vai trò, tối thiểu 5. Đây là phép thử tự xây dựng, chưa phải benchmark FITB chính thức của Polyvore. Các câu cùng bộ có tương quan và tập nam nhỏ.

Khi suy luận, vùng ảnh được mã hóa, lấy metadata đại diện từ món cùng vai trò gần nhất trong kho train. Phần thuộc tính tham chiếu này được ghi rõ trong giải thích. Mỗi vai trò thiếu lấy tối đa 20 ứng viên, tạo tối đa 400 bộ. RF chấm điểm bộ gồm vùng ảnh giữ lại và hai món bổ sung. Luật và cosine tạo điểm tổng hợp. MMR chọn tối đa 5 bộ dựa cả vector ảnh và mức lặp sản phẩm.

Bàn giao `ranking.json`, `ranking-report.json`, `recommendation.py`, `app.py` và giao diện.

## Thứ tự chạy đầy đủ

1. `prepare`.
2. `encoder` rồi `rules`.
3. `compatibility` rồi `ranking`.
4. `detector` sau khi có Fashionpedia.
5. `python -m fashion_ai.app`.

Mỗi checkpoint là đầu ra thật của một lệnh huấn luyện. Ứng dụng kiểm tra các tệp đầu ra trước khi suy luận. Khi đổi dữ liệu hoặc huấn luyện lại, dừng ứng dụng rồi khởi động lại để giải phóng mô hình cache.

## Kiểm tra

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm tra quy tắc lọc nam, tách split, biểu diễn bất biến thứ tự, âm tính tổng hợp, luật, MMR và xác thực phiên web. AP của detector cần được chạy trên Fashionpedia thật.

## Phân công báo cáo

Mỗi người nộp mã phụ trách, artifact, báo cáo JSON và phần viết về dữ liệu đầu vào, thiết lập thuật toán, kết quả, baseline và giới hạn. Các báo cáo được đọc trực tiếp ở phần Đánh giá trên website. Mã dùng torch/torchvision cho thị giác, mlxtend cho luật, scikit-learn cho RF/LR và Flask cho ứng dụng cục bộ.

## Trạng thái lần chạy ban đầu

Đã chạy encoder 2 epoch trên CPU, trích xuất embedding, khai phá 206 luật, huấn luyện RF/LR và chọn cấu hình MMR. Có 1.199 sản phẩm sau lọc và tách split. 12 kiểm thử đã chạy thành công, gồm tiếp tục huấn luyện đúng batch, luồng tải ảnh, khoanh vùng thủ công và trả bộ phối bằng checkpoint thật.

RF đạt ROC-AUC khoảng 0,535 trên test âm tính tổng hợp; Hit@5 của phép thử xếp hạng khoảng 0,274. Đây là kết quả ban đầu còn thấp, cần cải thiện dữ liệu và huấn luyện trước khi kết luận chất lượng. Chưa có checkpoint Fashionpedia: chức năng tự động phân đoạn đang chờ huấn luyện. Hiện có thể dùng ảnh tự khoanh vùng hoặc chọn món trong kho để chạy các bước gợi ý.
