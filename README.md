# FORM — Gợi ý phối đồ nam với Stylique/Polyvore

Dự án chỉ dùng https://huggingface.co/datasets/Stylique/Polyvore. Dataset GitHub xthan cũ và các artifact học từ bộ đó đã được gỡ.

## Chạy website

Cần Node.js 20 trở lên, không npm install và không API khi sử dụng:

```powershell
node scripts/serve.mjs
```

Mở http://localhost:3000. Nếu metadata mới chưa tải đủ, website thông báo chờ dữ liệu thay vì hiển thị kết quả cũ.

## Tải và chuẩn bị dữ liệu

Dataset tại `datasets/Stylique-Polyvore`. Theo dõi `.download/download-report.json` và `TRANG-THAI-TAI.md` trong thư mục đó. Tổng bản nguồn gồm 35 tệp khoảng 17 GB; ảnh `images.zip` 2,52 GB. Tải nền vẫn giữ nguyên.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-polyvore-download.ps1
node scripts/train.mjs
```

Train cần metadata đầy đủ, SHA-256 khớp manifest nguồn. Không đọc các tệp .part/.tail và không quay về dataset cũ nếu nguồn mới chưa sẵn sàng. Học xong, tải lại trình duyệt.

Sau khi `images.zip` tải đủ:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/import-stylique-images.ps1
```

Lệnh xác minh SHA-256 ảnh ZIP, trích ảnh của danh mục nam vào `web/images/<item_id>.jpg` và tạo chỉ mục ảnh. Không ghép ảnh theo tên hoặc ID bộ cũ. Nếu chưa có ảnh, giao diện chỉ dùng minh họa loại đồ và ghi rõ.

## Cấu trúc nguồn mới

- `polyvore_item_metadata.json`: từ điển theo item_id. Có title/url_name, description, đường dẫn danh mục (khóa gốc viết là catgeories), category_id và semantic_category.
- `polyvore_outfit_titles.json`: tên bộ theo set_id.
- `categories.csv`: category_id, loại chi tiết, nhóm ngữ nghĩa; một ID có thể có nhiều loại chi tiết, không được coi một nhãn male trong số đó là nhãn nam cho toàn bộ ID.
- `disjoint/train.json`, `valid.json`, `test.json`: bộ có set_id và danh sách item_id/index. Các split được giữ nguyên; chỉ học TF-IDF, K-Means và kho KNN từ train.
- `nondisjoint` và `maryland_polyvore_hardneg` vẫn được tải đầy đủ, nhưng không được trộn vào mô hình đang dùng split disjoint.

## Lọc đồ nam

Ưu tiên loại metadata có Women's/Female trong đường dẫn danh mục hoặc trường Gender: Female trong mô tả. Giữ sản phẩm có một trong các bằng chứng: đường dẫn danh mục nam; Gender: Male trong mô tả; hoặc category_id chỉ có các nhãn fine-grained bắt đầu bằng male trong categories.csv. Đây là quy tắc lọc metadata, không phải nhãn giới tính độc lập được xác thực; ghi bằng chứng cho từng sản phẩm.

Không dùng substring men để khớp women's. Không suy luận giới tính từ khuôn mặt hoặc cơ thể. Sản phẩm không có bằng chứng nam hoặc không thuộc vai trò được hỗ trợ bị loại. Chỉ phần món nam được giữ từ một bộ; không khẳng định cả bộ gốc là đồ nam.

## K-Means, KNN, Cosine Similarity

TF-IDF dùng tên/nhóm sản phẩm, từ vựng chỉ học trên train. Vector món và vector gộp bộ được chuẩn hóa L2. K-Means Lloyd/Euclidean chia 8 cụm, tối đa 30 vòng, khởi tạo xác định. KNN truy hồi K bộ lân cận theo cosine (K chọn từ 5/10/20 trên valid). Điểm ứng viên bổ sung = 0,35 bằng chứng trong bộ lân cận + 0,40 tương đồng với món cùng vai trò trong bộ lân cận + 0,25 cosine truy vấn. Đây là điểm thiết kế, không phải xác suất phối đẹp.

Kiểm tra tự xây dựng: che một món nam trong bộ valid/test, thêm ba món khác cùng vai trò từ chính split giữ lại và xếp hạng bốn lựa chọn. TF-IDF, centroid và bộ KNN không học từ test. Báo cáo sau khi train: `data/model-report.json`; không dùng lại con số 52,4% hoặc 1.694 sản phẩm của dataset cũ. Phép thử không phải benchmark FITB chính thức hoặc chấm thẩm mỹ; các câu cùng bộ có thể tương quan.

## Kiểm tra

```powershell
node --test tests/core.test.cjs tests/stylique.test.cjs
```

Kiểm tra artifact được bỏ qua có thông báo nếu metadata chưa tải đủ; khi train hoàn tất, nó kiểm tra nguồn Stylique, bằng chứng lọc nam và bộ train disjoint.

## Nguồn nghiên cứu

README gốc trong dataset trích dẫn Vasileva, Plummer, Dusad, Rajpal, Kumar & Forsyth, *Learning Type-Aware Embeddings for Fashion Compatibility*, ECCV 2018. Giữ nguyên tài liệu nguồn và các điều kiện sử dụng đi kèm trong `datasets/Stylique-Polyvore/README`.
## Trạng thái chuyển đổi hiện tại

Metadata mới đã được xác minh SHA-256. Đã học lại 497 sản phẩm nam train, 69 bộ có ít nhất hai món nam; 7 kiểm thử đạt. Bài kiểm tra riêng có 257 câu test, chọn đúng 88 câu (34,2%), baseline cosine 33,1%, ngẫu nhiên 25%. Validation sau loại bộ trùng món train chỉ còn 8 câu, vì vậy lựa chọn K và chênh lệch kết quả chỉ mang tính tham khảo. Nguồn gắn tên disjoint nhưng kiểm tra phần nam vẫn phát hiện trùng train/valid; các bộ có món train được loại khỏi đánh giá. Không khẳng định split của bản mirror hoàn toàn item-disjoint.

Trình tải nền được ưu tiên images.zip. Khi ZIP tải đủ và kiểm tra SHA-256 xong, nó tự chạy script nhập ảnh cho website. Các tệp đặc trưng lớn vẫn tiếp tục được tải, không bỏ khỏi yêu cầu tải đầy đủ.

## Phạm vi repo GitHub

Repo chứa mã nguồn, danh mục sản phẩm đã xử lý trong web/catalog.js, mô hình và báo cáo đánh giá. Dataset Stylique/Polyvore gốc có tại https://huggingface.co/datasets/Stylique/Polyvore. Chạy scripts/index-images.mjs để lập lại chỉ mục sau khi có ảnh cục bộ.
