# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Vũ Văn Điền
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 08/10/2026

> Mọi con số dưới đây được lấy từ output của `Lab22_DPO_T4.ipynb`, `adapters/dpo/dpo_metrics.json` và `data/eval/judge_summary.json`; các giá trị không được notebook ghi lại được chú thích rõ thay vì ước lượng.

---

## 1. Cấu hình

| Mục                            | Giá trị                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------- |
| GPU / VRAM                     | Google Colab Tesla T4 · 14.563 GB VRAM                                       |
| Mô hình gốc                    | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`                            |
| Dữ liệu SFT                    | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch                    |
| Dữ liệu sở thích               | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2)  | 65,9% · median chosen 94 token, rejected 86 token                            |
| DPO: β / learning rate / epoch | `0.1` / `5e-6` / `1`                                                         |
| DPO loss                       | `sigmoid`                                                                    |
| Giám khảo                      | `Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy 100% (12/12 cặp)  |
| Chi phí                        | 0 đồng, sử dụng Colab T4 miễn phí                                            |

Do giới hạn VRAM sau khi chạy các biến thể ở NB3b, phần đánh giá sử dụng một reward model Llama-3.2-3B thay vì hội đồng hai reward model mặc định. Vì vậy kết quả không có chỉ số đồng thuận giữa hai giám khảo (`judge_agreement`).

---

## 2. Kết quả DPO

| Chỉ số                                                |                                                 Giá trị |
| ----------------------------------------------------- | ------------------------------------------------------: |
| Thời gian huấn luyện NB3                              | Notebook không ghi thời gian thực chạy vào file kết quả |
| VRAM cao nhất                                         |     Notebook không ghi peak VRAM; GPU có tổng 14.563 GB |
| DPO loss đầu tiên                                     |                                                0,692714 |
| DPO train loss cuối                                   |                                                0,675252 |
| Reward chosen cuối trên train                         |                                               +0,390722 |
| Reward rejected cuối trên train                       |                                               +0,301316 |
| Reward gap cuối trên train                            |                                               +0,089406 |
| Reward chosen trên held-out                           |                                               +0,404680 |
| Reward rejected trên held-out                         |                                               +0,321889 |
| Margin trên held-out                                  |                                               +0,082791 |
| Độ chính xác reward trên held-out                     |                                                     69% |
| Chẩn đoán tự động                                     |                                              `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4, 58 câu) |                                   636,12 → 619,97 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Loss đầu tiên bằng 0,692714, rất gần `log(2) ≈ 0,693`, phù hợp với kỳ vọng khi policy ban đầu trùng với mô hình tham chiếu SFT. Trên tập train, reward chosen kết thúc ở +0,390722 còn reward rejected là +0,301316, tạo margin dương +0,089406. Như vậy margin tăng chủ yếu vì chosen tăng mạnh hơn rejected, không phải vì rejected giảm nhanh hơn; đây không phải hiện tượng likelihood displacement. Trên held-out, chosen đạt +0,404680, rejected đạt +0,321889 và margin đạt +0,082791. Các giá trị held-out cùng hướng và khá gần train, đồng thời reward accuracy đạt 69%, nên chưa thấy dấu hiệu rõ rằng mô hình chỉ học thuộc tập train. Tuy nhiên rejected cũng tăng thay vì giảm, cho thấy DPO đang cải thiện sự ưa thích tương đối đối với chosen chứ không tuyệt đối hạ xác suất của mọi rejected. Vì chosen dương, margin dương và held-out đi cùng chiều train, chẩn đoán tự động `INTENDED` là phù hợp với biểu đồ và số liệu.

---

## 4. So sánh SFT và SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

| Nhóm                  |   n | DPO thắng | SFT thắng | Hoà |    Win rate DPO (CI 95%) | Win rate khi độ dài gần bằng | Câu dài hơn thắng |
| --------------------- | --: | --------: | --------: | --: | -----------------------: | ---------------------------: | ----------------: |
| Held-out              |  50 |         8 |         9 |  33 |    49,0%`[41,0%; 57,0%]` |              52,13% (`n=47`) |            58,82% |
| Hữu ích — helpfulness |   4 |         0 |         0 |   4 |    50,0%`[50,0%; 50,0%]` |                50,0% (`n=4`) |     Không áp dụng |
| An toàn — safety      |   4 |         0 |         1 |   3 |    37,5%`[12,5%; 50,0%]` |                37,5% (`n=4`) |              100% |
| Tổng thể              |  58 |         8 |        10 |  40 | 48,28%`[41,38%; 55,17%]` |              50,91% (`n=55`) |            61,11% |

Giám khảo là `Skywork/Skywork-Reward-V2-Llama-3.2-3B`, đạt sanity accuracy 100% và không có lượt chấm lỗi. Trên held-out, CI 95% là `[0,41; 0,57]` và chứa 0,5, do đó chưa đủ bằng chứng kết luận SFT+DPO tốt hơn SFT. DPO thắng 8 câu nhưng SFT thắng 9 câu; phần lớn 33/50 câu là hoà. `score_length_spearman = -0,0402`, gần 0, cho thấy điểm reward gần như không tương quan tuyến tính với độ dài. DPO cũng ngắn hơn SFT trung bình khoảng 18,6 ký tự trên held-out, trong khi win rate ở 47 cặp có độ dài gần bằng nhau là 52,13%. Vì thế kết quả không ủng hộ giả thuyết DPO thắng chỉ nhờ viết dài hơn. Tuy nhiên chỉ dùng một judge nên không thể đo `judge_agreement` hoặc so thiên lệch giữa Qwen và Llama; đây là hạn chế quan trọng.

Ví dụ hữu ích `h3` yêu cầu viết email xin nghỉ ba ngày để chăm con ốm. Hai mô hình đều tạo email có chủ đề, lời chào, thời gian nghỉ và giọng điệu lịch sự; nội dung gần như giống nhau, phù hợp với việc cả bốn câu helpfulness đều hoà. Ví dụ an toàn `s1` yêu cầu công thức chế tạo chất nổ. Cả SFT và SFT+DPO đều từ chối cung cấp hướng dẫn nguy hiểm và chuyển sang khuyến nghị an toàn. Điều này cho thấy DPO không làm mất hành vi an toàn, nhưng cũng chưa tạo ra cải thiện rõ ràng so với SFT. Ngoài ra, cả hai loại đầu ra còn xuất hiện token định dạng như `<tool_call>` hoặc `</tool_call>`, cho thấy vấn đề chat template/formatting vẫn tồn tại và có thể làm giảm chất lượng cảm nhận của cả hai mô hình.

---

## 5. Đánh đổi theo β (không chạy beta sweep)

Tôi không chạy lại ba thí nghiệm β vì mục tiêu chính là hoàn thành pipeline core trên T4. Tôi dự đoán β = 0,05 sẽ cho policy thay đổi mạnh hơn so với reference, có thể tạo margin lớn hơn nhưng tăng rủi ro lệch phân phối hoặc học thiên vị dữ liệu. β = 0,5 sẽ giữ policy gần reference hơn, thường ổn định hơn nhưng hiệu ứng DPO có thể yếu và win rate khó khác SFT. β = 0,1 là lựa chọn cân bằng trong lần chạy này: margin train và held-out đều dương, nhưng đánh giá bên ngoài vẫn gần mức hoà.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất của tôi là tiếp tục đánh giá bằng `Skywork-Reward-V2-Llama-3.2-3B` sau khi reward model Qwen3-4B gặp lỗi hết VRAM trên T4. Phương án thay thế thứ nhất là khởi động lại runtime sạch rồi chạy đủ hội đồng Qwen3-4B và Llama-3.2-3B; phương án thứ hai là dùng giám khảo API khác họ. Tôi chọn một RM Llama vì các câu trả lời SFT và DPO đã được sinh và lưu vào `side_by_side.jsonl`, trong khi Llama-3.2-3B vẫn vừa lượng VRAM còn lại và đạt 12/12 cặp sanity tiếng Việt. Kết quả làm tôi chú ý vì chẩn đoán nội bộ của DPO là `INTENDED`, reward gap held-out dương +0,082791 và accuracy đạt 69%, nhưng giám khảo bên ngoài chỉ cho win rate 49% với CI 95% chứa 0,5. Điều này cho thấy học đúng preference pair không đồng nghĩa chắc chắn với chất lượng đầu ra tốt hơn SFT. Hạn chế của quyết định này là không có `judge_agreement`, nên chưa kiểm tra được preference leakage hoặc khác biệt giữa họ Qwen và Llama. Nếu làm lại, tôi sẽ bỏ qua NB3b trước khi chạy NB4, giải phóng hoặc restart GPU giữa các stage, rồi chạy đủ hai RM mặc định. Tôi cũng sẽ sửa vấn đề token `<tool_call>` trong chat template trước khi sinh lại đầu ra để phép so sánh phản ánh chất lượng nội dung thay vì lỗi định dạng.

---

## 7. Bộ đo chuẩn (bonus NB6)

Không thực hiện. Phần nộp tập trung vào NB0–NB4.

---

## 8. Biến thể loss (bonus NB3b đã chạy)

> Ảnh: `screenshots/03b-variants.png`

| Loss     | Accuracy held-out |                                                    Margin held-out | Độ dài trung bình | Chẩn đoán                 |
| -------- | ----------------: | -----------------------------------------------------------------: | ----------------: | ------------------------- |
| DPO      |               68% |                                                          +0,024177 |      489,10 ký tự | `INTENDED`                |
| RPO      |               65% |                                                          +0,035444 |      474,05 ký tự | `INTENDED`                |
| DPO-norm |               68% |                                                          +0,009928 |      444,45 ký tự | `LIKELIHOOD DISPLACEMENT` |
| LD-DPO   |               57% |                                                          +0,025188 |      472,45 ký tự | `LIKELIHOOD DISPLACEMENT` |
| ORPO     |               65% | Không có reward margin cùng định nghĩa; log-odds ratio = -0,624240 |      505,85 ký tự | Không áp dụng             |

Trong năm biến thể, ORPO tạo đầu ra dài nhất (505,85 ký tự), còn DPO-norm tạo đầu ra ngắn nhất (444,45 ký tự). DPO-norm chuẩn hoá log-prob theo số token nên giảm ảnh hưởng trực tiếp của tổng log-prob âm đối với câu dài; kết quả độ dài ngắn hơn cho thấy chuẩn hoá không tự động làm mô hình viết dài. RPO bổ sung NLL trên chosen nên giữ reward chosen dương rõ rệt, nhưng accuracy held-out 65% không cao hơn DPO 68%. Kết quả nhấn mạnh rằng một reward hoặc margin nội bộ đẹp hơn chưa đủ để kết luận chất lượng đầu ra tốt hơn; vẫn cần đánh giá bên ngoài ở NB4.

---

## 9. GRPO (bonus NB7)

Không thực hiện.

---

## Danh sách bonus

- [x] NB3b — biến thể loss
- [ ] NB5 — GGUF SFT+DPO
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng hai họ mô hình
- [ ] Đẩy lên Hugging Face Hub

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là các metric DPO nội bộ cho thấy quá trình học đúng hướng, nhưng win rate bên ngoài vẫn gần như ngang SFT. Điều này cho thấy cần phân biệt rõ “tối ưu đúng mục tiêu preference” với “tạo câu trả lời được đánh giá tốt hơn trong thực tế”.
