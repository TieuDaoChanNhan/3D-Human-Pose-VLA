# PAB-Spline VLA / Omni-Modal Project — Tổng quan dự án

*Cập nhật: 26/07/2026. Tài liệu này dành cho team nội bộ và leadership (Huu, Van Khue, cộng sự kỹ thuật) — có dùng thuật ngữ kỹ thuật nhưng cố gắng giải thích rõ cho người không trực tiếp code hằng ngày.*

---

# PHẦN 1 — TỔNG QUAN NHANH

## 1. Dự án là gì

Dự án xây dựng một **mô hình ngôn ngữ lớn (LLM) học đa phương thức (omni-modal)**: một mạng nơ-ron duy nhất (backbone Qwen3) có thể đọc/hiểu và sinh ra nhiều loại "phương thức" (modality) khác nhau — video, ảnh tĩnh, âm thanh, chuyển động 3D của cơ thể người (pose/action), và văn bản — bằng cách biến TẤT CẢ các loại dữ liệu này thành **token** (giống hệt cách LLM xử lý từ ngữ) rồi xen kẽ chúng trong cùng một chuỗi huấn luyện.

Tên gọi "PAB-Spline VLA" (Pose-Action-Behavior Spline, Vision-Language-Action) là tên ban đầu khi dự án còn tập trung hẹp vào "huấn luyện robot hình người (humanoid) qua video con người". Tên đó vẫn được giữ cho nhánh video+pose (nhánh lớn nhất, dùng lâu nhất), nhưng **phạm vi thực tế của dự án đã mở rộng** — xem mục 2.

## 2. Động lực & vì sao đổi hướng (scope pivot)

Ban đầu, mục tiêu là: dùng ~40,000 video YouTube (bộ FineVideo) để trích xuất chuyển động 3D người thật, tokenize thành "agent token", rồi huấn luyện 1 LLM sinh ra hành động hợp lý từ mô tả bằng lời — hướng tới việc dạy robot hình người bắt chước con người.

Ngày 20/07/2026, Huu (leader dự án) xác nhận trực tiếp phạm vi thực sự rộng hơn nhiều:

> *"omni means all modes: image, video, sound, action, imu, etc. as long as we balance the dataset and create cross-modal bindings."*

Nói cách khác, tiêu chí chấp nhận 1 nguồn dữ liệu mới không phải "có video/pose/action hay không", mà là:
1. **License permissive** (cho phép dùng lại, không vướng bản quyền) — nguyên tắc cứng, không thương lượng lại.
2. **Cân bằng tỷ trọng modal** — không để 1 loại dữ liệu áp đảo, làm loãng các modal khác.
3. **Tạo được cross-modal binding thật** — tức là dữ liệu phải giúp model học mối liên hệ THẬT giữa các modal (ví dụ: audio khớp đúng với hình ảnh/hành động đang diễn ra), không chỉ là nhét chung nhiều loại dữ liệu không liên quan.

Nhánh video+pose (FineVideo/OmniVideo) vẫn là nhánh lớn nhất và phát triển nhất, nhưng không còn là toàn bộ phạm vi — dự án đã thêm ảnh tĩnh+caption tổng hợp (`synth_llava`), âm thanh tổng hợp+text (`laion/emotional-roleplay`), và dữ liệu audio+text quy mô lớn (`MV-Omni`).

## 3. Kiến trúc tổng quan — "mọi thứ đều là token"

Ý tưởng cốt lõi (giống cách AnyGPT — 1 hệ thống tương tự đã công bố — làm, nhưng đi xa hơn ở phần "action"):

- Mỗi modal có 1 tokenizer riêng biến dữ liệu gốc thành chuỗi số nguyên (token id):
  - **Seed2** — ảnh tĩnh/keyframe → 8,192 token khả dĩ
  - **Cosmos** — đoạn video ngắn (spatial-temporal) → 64,000 token khả dĩ
  - **Agent** — chuyển động 3D người (17 khớp xương, chuẩn H36M) → token toạ độ x/y/z + thời gian, mã hoá bằng PCHIP thích ứng (xem mục kỹ thuật ở Phần 2)
  - **SNAC / Listen / Speak** — âm thanh → token rời rạc từ bộ mã hoá âm thanh SNAC (Orpheus)
  - **Text** — văn bản thường (caption, câu hỏi, hội thoại)
- Tất cả các họ token này được **gộp vào cùng 1 vocab** của 1 tokenizer duy nhất (hiện tại: dựa trên Qwen3, ~274,561 token thật) — mỗi token VLA (ví dụ `<seed2_1137>`, `<pelvis_x_128>`) là **1 token nguyên tử** (atomic), không bị tách nhỏ như BPE thường làm với từ lạ.
- Một bản ghi huấn luyện là 1 chuỗi text xen kẽ nhiều khối modal, mỗi khối được bọc bởi cặp thẻ mở/đóng riêng (`<seed2>...</seed2>`, `<cosmos>...</cosmos>`, `<agent>...</agent>`, `<listen>...</listen>`, `<caption>...</caption>`...) — cho model tín hiệu rõ ràng "khối này kết thúc, chuyển modal khác".
- Backbone hiện tại: **Qwen3 1.7B** (thực tế ~1.97B tham số sau khi cộng thêm embedding cho vocab mở rộng), huấn luyện trên cụm JUPITER (GH200 GPU) bằng Megatron-LM.

## 4. Nguồn dữ liệu (tính đến 26/07/2026)

| Nguồn | Modal chính | Quy mô (token thật, mix mới nhất) | Vai trò |
|---|---|---|---|
| **FineVideo-VLA** | video + pose + audio + text | ~10.9B | Nhánh flagship — video YouTube thật + pose 3D trích xuất |
| **OmniVideo-100K** | video + pose (subset) + audio + QA | ~1.98B | Video + câu hỏi-đáp đa phương thức có sẵn |
| **Harmony4D** | pose (chuyển động 2 người tương tác gần) | ~0.32B (oversample 20x) | Bù lỗ hổng che khuất/multi-person mà video đơn mắt (FineVideo) không có |
| **MV-Omni** (MixtureVitae-Omni) | audio + ảnh + text | ~20.4B | Nguồn lớn nhất — chủ lực cho "language backbone", không có pose/video |
| **synth_llava / synth_llava2** | ảnh tĩnh + caption | ~0.1B | Ảnh tổng hợp+caption, dữ liệu riêng của Huu |
| **laion/emotional-roleplay** | audio (giọng nói tổng hợp) + text | ~0.11B | Audio↔text, dạy model vai trò "nghe" (`<listen>`) và "nói" (`<speak>`) |

Toàn bộ 6 nguồn này vừa được rebuild lại theo kiến trúc **window=8** (mỗi đoạn pose/video dài 8 khung hình) và định dạng audio mới (`<listen>`/`<speak>` thay vì `<snac>` trần) — tổng **33.83 tỷ token thật**, publish công khai trên HuggingFace (`EmpathicRobotics` org).

## 5. Lịch sử training nổi bật (v1 → v6)

| Bản | Kiến trúc | Test PPL (càng thấp càng tốt) | Ghi chú |
|---|---|---|---|
| v1/v2 | window=8, GPT-NeoX rồi Qwen3 | **v2: 5.77** | v2 vẫn là baseline mạnh nhất project từng đạt |
| v3 | window=24, đổi nhiều biến cùng lúc | 27.58 | Thụt lùi rõ rệt |
| v4 | window=24, sửa 2 bug (drop_cosmos, doc-packing) | 15.78 | Đỡ hơn v3 nhưng vẫn kém xa v2 |
| v5 | window=24, giảm drop_cosmos | 16.75 | Không cải thiện PPL, chỉ đổi hành vi cosmos-continuity |
| **v6** (seq4096/8192/16384) | **window=8 rebuild** (`w8_new` mix, giống kiến trúc v2 + thêm Harmony4D/MV-Omni/roleplay) | **5.98 – 6.36** | Áp sát v2, phục hồi gần hết chất lượng đã mất ở v3-v5 |

**Bài học lớn nhất**: vấn đề của v3/v4/v5 không phải là "window=24 tự nó xấu", mà là **quá nhiều biến thay đổi cùng lúc** (window size, mix nguồn, dropout, token-cost) khiến không tách bạch được nguyên nhân. v6 quay lại đúng công thức đã chứng minh hiệu quả của v2 (window=8) và chỉ thêm dữ liệu mới — kết quả PPL hồi phục gần như hoàn toàn.

## 6. Phát hiện quan trọng nhất: gap "instruction-following"

Xuyên suốt **mọi version đã test (v2 → v6, không sót bản nào)**: khi cho model 1 mô tả hoàn toàn mới (chưa từng thấy trong training, ví dụ "A person is running in a park"), **caption mà model tự sinh ra luôn sai chủ đề** (ví dụ bịa ra "riding a mountain bike", "riding a slide"...) — mô hình chỉ giỏi TÁI TẠO ĐÚNG những gì đã thấy y hệt trong dữ liệu training, chứ chưa học được cách suy luận/điều kiện hoá theo văn bản mới lạ.

Đây là gap **duy nhất chưa version nào sửa được** dù đã thử rất nhiều biến pretraining khác nhau (window size, dropout, seq_length, mix nguồn). Nguyên nhân được chẩn đoán qua so sánh với paper AnyGPT: dự án thiếu hẳn 1 loại dữ liệu — **instruction-tuning data** (hội thoại đa dạng dạy model "điều kiện hoá đúng theo yêu cầu mới"), giống bộ AnyInstruct-108k mà AnyGPT dùng.

## 7. Trạng thái hiện tại

- **Model public duy nhất**: `EmpathicRobotics/vla-1.7b-qwen3-v2` (bản tốt nhất, v3-v6 vẫn chỉ là checkpoint nội bộ, chưa publish).
- **6 dataset đã/đang public lên HF** (bản mới nhất, window=8 + listen/speak): `FineVideo-Phase7-Flattened`, `harmony4d-flattened`, `omnivideo-100k-final`, `MV-Omni`, `synth-llava`, `emotional-roleplay-finetuning-dataset-flattened`.
- **Tokenizer**: `EmpathicRobotics/tokenizer-vla-qwen3-v2` (274,561 vocab thật) là bản khuyến nghị hiện tại — hỗ trợ đủ mọi token modal + `<listen>`/`<speak>`.
- Đã có bộ eval script tái sử dụng được (sanity/atomicity, temporal-continuity, full-chain text-to-media) để so sánh khách quan giữa các version.

## 8. Hướng tiếp theo (roadmap ngắn)

1. **Ưu tiên cao nhất**: xây "VLA-Instruct" — dữ liệu instruction-tuning dạng SFT stage riêng (retrieval-based: LLM sinh kịch bản hội thoại đa dạng → thay placeholder bằng token thật khớp ngữ nghĩa từ dữ liệu đã có), nhắm thẳng vào gap instruction-following.
2. Chốt 1 seq_length cho v6 (seq8192 đang nhỉnh nhất) sau khi verify đa-seed.
3. Cân nhắc tích hợp DROID (robot action thật, Apache 2.0) — cần thiết kế vocab action 7-DoF riêng, ưu tiên thấp hơn #1.
4. Duy trì kỷ luật license/eval khi mở rộng thêm nguồn dữ liệu mới.

---

# PHẦN 2 — CHI TIẾT ĐẦY ĐỦ

## 2.1. Bối cảnh & động lực đầy đủ

### Giai đoạn 1 — "VLA cho humanoid" (trước 20/07/2026)

Mục tiêu ban đầu: dùng video YouTube công khai (bộ **FineVideo**, ~40,000 video, do HuggingFaceFV công bố) để:
1. Trích xuất chuyển động 3D người (pose) từ video 2D thông thường (không cần camera đặc biệt).
2. Tokenize chuyển động đó thành "agent token" — một dạng "ngôn ngữ hành động" mà LLM có thể sinh ra như sinh văn bản.
3. Kết hợp với token hình ảnh (Seed2), token video (Cosmos), token âm thanh, và caption/text, huấn luyện 1 LLM duy nhất hiểu và sinh được cả video lẫn hành động từ mô tả bằng lời.

Ý tưởng nền: nếu 1 LLM có thể "nói" ra chuyển động hợp lý của con người từ hàng chục nghìn giờ video thật (rẻ hơn nhiều so với thu thập dữ liệu robot thật), đó là 1 con đường khả thi để huấn luyện robot hình người (humanoid) mô phỏng hành vi con người, dùng closed-loop simulation làm thước đo cuối cùng.

### Giai đoạn 2 — Pivot sang omni-modal (20/07/2026)

Huu xác nhận trực tiếp qua Discord: phạm vi thật của dự án là **omni-modal** — bind bất kỳ tổ hợp modal nào (ảnh, video, âm thanh, action, IMU...), không bắt buộc phải liên quan tới robot/hành động con người. Tiêu chí chấp nhận 1 nguồn dữ liệu mới: (1) license permissive, (2) cân bằng tỷ trọng modal trong tổng dataset, (3) tạo được cross-modal binding thật (không phải trộn thô nhiều loại dữ liệu không liên quan).

Ví dụ 2 nguồn được thêm vào đúng theo tiêu chí mới (không có video/pose/action nào cả):
- `synth_llava`/`synth_llava2` (`mixture-vitae-backup/MixtureVitae-Backup`) — ~604K cặp ảnh+caption tổng hợp, dữ liệu riêng của Huu, dùng để dạy token `<seed2_N>` (vì Seed2 là tokenizer duy nhất chấp nhận ảnh đơn lẻ, không cần video).
- `laion/emotional-roleplay-finetuning-dataset` — 67,491 đoạn giọng nói tổng hợp (TTS) + text, dùng cho token SNAC/audio.

Nhánh FineVideo/OmniVideo (video+pose+action) vẫn là nhánh lớn nhất và có framing "VLA cho humanoid" ban đầu — nhưng không còn là toàn bộ scope. Có 1 lo ngại đã được ghi nhận nội bộ: khi scope dữ liệu mở rộng nhanh, kỷ luật về eval protocol/nghiên cứu cần theo kịp, tránh tích luỹ "nợ eval" (thêm nguồn mới mà chưa có cách đo chất lượng tương ứng).

## 2.2. Kiến trúc hệ thống chi tiết

### 2.2.1. Token hoá từng modal

| Modal | Tokenizer | Vocab size | Ghi chú |
|---|---|---|---|
| Seed2 (ảnh/keyframe) | Seed2Tokenizer (dựa trên Stable Diffusion 2.1-unclip) | 8,192 | 1 khung hình → 32 token, giữ 100% (không dropout) |
| Cosmos (video ngắn) | NVIDIA Cosmos tokenizer | 64,000 | Mỗi đoạn 8 khung hình → 200 token (window=8) hoặc 896 token (window=24, đã bỏ dùng); giữ 50% (dropout ngẫu nhiên theo chunk để cân bằng tỷ trọng) |
| Agent (pose 3D) | Tự thiết kế — Adaptive PCHIP | biến đổi theo 17 khớp × (t, x, y, z) | Xem chi tiết mục 2.2.3 |
| SNAC/Listen/Speak (âm thanh) | SNAC (Orpheus, `hubertsiuzdak/snac_24khz`) | 3 dải: L0 (128266-132361), L1a (132362-136457), L1b (144650-148745), tổng 12,290 token | `<listen>` = nghe (input), `<speak>` = nói (output do model tự sinh) — phân biệt vai trò, không phải phân biệt định dạng |
| Text (caption/speech/hội thoại) | BPE thường (Qwen3 base) | phần còn lại của vocab | Không cần vocab riêng — text tự nhiên dùng chung BPE |

Tất cả token modal được đăng ký vào tokenizer bằng `add_tokens(special_tokens=True)` — đảm bảo **atomic** (không bị BPE tách nhỏ). Đây là 1 bug quan trọng đã sửa từ rất sớm: nếu không đăng ký đúng cách, `<seed2_1137>` sẽ bị tách thành 7 mảnh BPE rời rạc — mô hình đầu tiên của project (`vla-1.7b-pab-spline-25b-test`) bị lỗi này.

### 2.2.2. Lịch sử tokenizer & vocab

| Tokenizer | Base | Vocab | Dùng cho |
|---|---|---|---|
| `tokenizer-vla-adaptive` | GPT-NeoX-20b | 144,215 | v1 — chỉ có agent token, chưa có SNAC |
| `tokenizer-vla-adaptive-v2` | GPT-NeoX-20b | 156,509 | v3-v6 (bản GPT-NeoX) — thêm SNAC (bare `<snac>`) + caption/speech |
| `tokenizer-vla-qwen3` | Qwen3 | 257,901 | Bản Qwen3 đầu tiên, dùng cho model v2 |
| **`tokenizer-vla-qwen3-v2`** | Qwen3 | **274,561** (đệm 274,688) | **Bản khuyến nghị hiện tại** — thêm `<listen>`/`<speak>` wrapper (thay `<snac>` trần), 2 dải SNAC mới phát hiện qua stream dữ liệu thật |

### 2.2.3. Định dạng token Agent (pose 3D) — Adaptive PCHIP

Mỗi cửa sổ thời gian (8 hoặc 24 khung hình) của 1 người trong video được mã hoá thành chuỗi token theo 17 khớp xương chuẩn H36M (pelvis, hông, gối, mắt cá, cột sống, vai, khuỷu tay, cổ tay, đầu...):

```
<fps_30>
<pelvis> <pelvis_t_0> <pelvis_x_128> <pelvis_y_128> <pelvis_z_128>
         <pelvis_t_7> <pelvis_x_130> <pelvis_y_128> <pelvis_z_130> </pelvis>
<r_hip>  <r_hip_t_0>  <r_hip_x_140> ...  </r_hip>
...17 khớp...
```

- Mỗi khớp được biểu diễn bằng **2, 4, hoặc 8 điểm điều khiển (control point)** tuỳ theo độ cong của chuyển động (khớp gần như đứng yên → 2 điểm; chuyển động phức tạp → 8 điểm) — đây là ý tưởng "adaptive" (thích ứng), tiết kiệm token cho chuyển động đơn giản.
- Toạ độ x/y/z được lượng tử hoá về số nguyên 0-255 (uint8), ánh xạ khoảng [-2.0m, +2.0m] quanh gốc (root-centred tại pelvis).
- Để khôi phục lại đủ 8 (hoặc 24) khung hình gốc từ các điểm điều khiển thưa, dùng nội suy **PCHIP** (Piecewise Cubic Hermite Interpolating Polynomial) — mượt hơn nội suy tuyến tính, không bị "overshoot" như spline bậc 3 thường.

### 2.2.4. Định dạng chuỗi & lịch sử window=8 vs window=24

Một bản ghi huấn luyện = 1 chuỗi text bắt đầu bằng header (`### Title:`, `### Context:`, `### Keywords:`, `### Speech:`), theo sau là các khối modal xen kẽ theo thứ tự thời gian, mỗi 8 (hoặc 24) khung hình 1 lần:

```
chunk 0: [caption?] [seed2?] [cosmos] [agent?] [listen?] [speech?]
chunk 1:            [cosmos] [agent?] [listen?]
...
```

Dự án từng chuyển từ window=8 sang window=24 (giữa năm 2026) với kỳ vọng "cửa sổ dài hơn = model nhìn được ngữ cảnh dài hơn". Tuy nhiên, kết quả thực nghiệm (xem mục 2.4) cho thấy **window=8 (bản v2) vượt trội hơn mọi bản window=24** trên mọi chỉ số đo được — không phải vì window=24 tệ về bản chất, mà vì việc chuyển sang window=24 đi kèm quá nhiều thay đổi cùng lúc (cosmos tốn gấp 4.5x token/chunk → buộc phải dropout mạnh hơn, corpus nhỏ hơn, thêm wrapper mới...). Vì vậy, **v6 (26/07/2026) quay lại window=8** làm kiến trúc chuẩn, chỉ thêm dữ liệu mới trên nền công thức đã chứng minh hiệu quả.

## 2.3. Data pipeline đầy đủ theo từng nguồn

### FineVideo-VLA (nhánh flagship, video+pose)
- **Nguồn gốc**: ~40,000 video YouTube từ bộ [FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) (HuggingFaceFV), license Apache 2.0 tự chứa nội dung.
- **Pipeline**: Step A (tokenize video → seed2/cosmos) → Phase 1 (HRNet phát hiện khớp 2D) → Phase 2 (MotionBERT nâng 2D→3D) → Phase 3 (chuẩn hoá động học/lọc nhiễu) → Phase 4 (lọc theo phát hiện người YOLO) → Phase 5 (tokenize agent adaptive PCHIP) → Phase 6 (merge agent+audio vào chuỗi video) → Phase 7 (flatten thành JSONL phẳng cho Megatron).
- **Quy mô hiện tại (v7, window=8, listen/speak)**: 371,892 bản ghi, 10,926,767,551 token thật.
- **HF repo**: `EmpathicRobotics/FineVideo-Phase7-Flattened` (dataset cuối), cộng các repo trung gian (`FineVideo-Prototype-Tokenized`, `FineVideo-Phase2-3DPose`, `FineVideo-Phase4-YOLOPose`, `FineVideo-Phase5-AgentTokens`).

### OmniVideo-100K
- **Nguồn gốc**: [MiG-NJU/OmniVideo-100K](https://huggingface.co/datasets/MiG-NJU/OmniVideo-100K), Apache 2.0, 52.9GB thật.
- **Nội dung**: video + QA đa phương thức có sẵn (99,983 cặp hỏi-đáp: 70,017 mở + 29,966 trắc nghiệm); pose (agent token) chỉ chạy trên 1 tập con thể thao (~15% video có agent).
- **Quy mô hiện tại (window=8)**: 5,214 bản ghi (mỗi bản ghi = 1 video), 1,979,126,756 token thật.
- **HF repo**: `EmpathicRobotics/omnivideo-100k-final`.

### Harmony4D
- **Nguồn gốc**: [Harmony4D](https://jyuntins.github.io/harmony4d/) — motion capture đa camera, ghi lại tương tác gần giữa 2 người (ôm, võ thuật, đấu kiếm, khiêu vũ).
- **Vai trò**: bù lỗ hổng mà pipeline FineVideo (video 1 camera thông thường) không giải quyết được — che khuất (occlusion) và tương tác nhiều người. FineVideo phải loại bỏ ~56% cửa sổ do bộ lọc che khuất/ảo giác; Harmony4D là dữ liệu ground-truth đa camera, 416/416 track qua lọc sạch (không cần các bộ lọc dành riêng cho lỗi ước lượng đơn mắt).
- **Oversample**: 20x (416 track vật lý quá nhỏ so với phần còn lại của mix, nên nhân lên 20 lần theo quyết định của Van Khue) — có 1 bug thật đã tìm và sửa: Megatron tự động lặp lại nội bộ bất kỳ nguồn nào có tỷ trọng cao hơn nội dung vật lý (`_get_num_epochs`), khiến oversample thật sự lên tới ~30.7x thay vì 20x dự định; đã sửa bằng cách cho Harmony4D 1 tỷ trọng riêng (đúng bằng tỷ lệ vật lý của nó), không theo công thức chia đều bucket.
- **Quy mô hiện tại (window=8)**: 8,320 bản ghi (416 track × 20x), 315,545,360 token thật.
- **HF repo**: `EmpathicRobotics/harmony4d-flattened`.

### MV-Omni (MixtureVitae-Omni)
- **Nguồn gốc**: HF `mixture-vitae/MixtureVitae-Omni` (`valid_snac` split) — 1,593,301 bản ghi thật (đã verify bằng cách giải nén toàn bộ 26/07/2026, sửa lại ước tính cũ "~1.78M" chưa từng đếm thật).
- **Nội dung**: audio (SNAC) + ảnh (Seed2, đã convert từ `<seed_N>` gốc) + text dạng câu hỏi ("Q: Listen to this and tell me what you heard..."). **Không có pose/cosmos**.
- **Vai trò**: nguồn lớn nhất trong toàn bộ mix (39.58% tỷ trọng của mix `w8_new`) — chủ lực cho "language backbone" và audio↔text binding.
- **Quy mô (token thật, dùng trong v6)**: 20,389,561,883 token.
- **Lưu ý license**: không tìm thấy dòng xác nhận license rõ ràng trong tài liệu nội bộ (khác `synth_llava` đã có xác nhận trực tiếp từ Huu) — cả 2 org `mixture-vitae` và `mixture-vitae-backup` đều do leader dự án kiểm soát nên khả năng cao là dữ liệu của Huu, nhưng gắn tag `license: other` (không SPDX cụ thể) theo quyết định của Van Khue (26/07/2026), chờ xác nhận rõ hơn nếu cần.
- **HF repo**: `EmpathicRobotics/MV-Omni` (mới upload).

### synth_llava / synth_llava2
- **Nguồn gốc**: `mixture-vitae-backup/MixtureVitae-Backup/data/multimodal` — dữ liệu ảnh+caption tổng hợp (kiểu llava_pretrain) do chính Huu tạo, xác nhận permissive trực tiếp (2026-07-21).
- **Quy mô**: 603,999 bản ghi (56 shard `synth_llava` + 95 shard `synth_llava2`), 19,327,968 token seed2 (32 token/ảnh).
- **Vai trò**: bổ sung khối lượng token `<seed2_N>` thuần (ảnh tĩnh), không có video/action.
- **HF repo**: `EmpathicRobotics/synth-llava`.

### laion/emotional-roleplay-finetuning-dataset
- **Nguồn gốc**: [laion/emotional-roleplay-finetuning-dataset](https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset), license CC-BY-4.0.
- **Nội dung**: 67,459/67,491 bản ghi (32 dòng bị loại do `adherence_score` ngoài khoảng hợp lệ) — giọng nói tổng hợp TTS (MOSS-TTS-Local v1.5), đa ngôn ngữ (chủ yếu Đức), ~184 giờ audio, mã hoá SNAC dạng "speak" (7 token/frame 12.5Hz, đủ cả 3 tầng codebook — khác "listen"-format chỉ giữ 2 tầng).
- **Vai trò**: dạy model vai trò `<speak>` (model tự sinh giọng nói) — bổ trợ cho `<listen>` (nghe) đã có từ FineVideo/OmniVideo.
- **Quy mô**: 54,578,440 token SNAC.
- **HF repo**: `EmpathicRobotics/emotional-roleplay-finetuning-dataset-flattened`.

### Các nguồn đã khảo sát nhưng KHÔNG dùng (lý do license)

| Nguồn | Lý do loại |
|---|---|
| stera-10m | Không permissive (đồng thuận Huu + Van Khue, 18/07) |
| AgiBot World | CC BY-NC-SA 4.0 (NonCommercial) |
| Apple EgoDex | CC-BY-NC-ND (NonCommercial + No-Derivatives) — tiếc vì rất khớp use-case |
| Meta ego-1k / EgoBrain | FAIR Noncommercial / CC-BY-NC, EgoBrain còn lạc chủ đề (EEG) |
| JRDB-Pose3D | Non-commercial license |
| SenseNova-SI-8M | License ảnh gốc KHÔNG xác minh được (dù tag HF là Apache-2.0) — treo lại chưa dùng |
| Open X-Embodiment | Registry 55-60 dataset con, license KHÔNG đồng nhất — cần audit từng cái |
| MINT-1T-HTML (phần ảnh) | Chỉ có URL, không track được license nguồn ảnh gốc — đã bỏ phần ảnh, giữ phần text |

### Ứng viên đang cân nhắc (chưa tích hợp)
- **DROID** (`nvidia/Cosmos3-DROID`) — robot action thật, license OpenMDW-1.1 (cho phép thương mại), 707GB — cần thiết kế vocab action 7-DoF riêng trước khi tích hợp (khác hẳn agent-token 17-khớp người).
- **NVIDIA PhysicalAI-Robotics-GR00T-X-Embodiment-Sim** — CC-BY-4.0, 345K+ trajectory humanoid/robot-arm mô phỏng — ứng viên robot-action mạnh nhất mới tìm được.
- **IPEC-COMMUNITY/EO-Data1.5M** — Apache 2.0, 1.5M sample, đúng format interleaved vision-language-action, không merge trực tiếp được (action-space robot khác agent-token người) nhưng là template tốt nhất hiện có cho VLA-Instruct.

## 2.4. Lịch sử training v1 → v6 đầy đủ

| Bản | Ngày | Kiến trúc | Train iters | Test PPL | Bài học chính |
|---|---|---|---|---|---|
| v1 | ~Mar 2026 | window=8, GPT-NeoX, chỉ agent | — | — | Tokenizer bug (BPE tách nhỏ token VLA) — model đầu tiên bị lỗi này |
| v2 | Jun 2026 | window=8, Qwen3, 5 nguồn | 7,632 | **5.77** | Baseline mạnh nhất project từng đạt, chưa version nào vượt qua tới nay |
| v3 | 02/07/2026 | window=24, đổi window+wrapper+corpus+drop_cosmos cùng lúc | 881 | 27.58 | Thụt lùi nặng — quá nhiều biến đổi đồng thời, không tách bạch được nguyên nhân |
| v4 | 24/07/2026 | window=24, giữ nguyên v3 + train nhiều bước hơn | ~2032+ | 15.78 | Đỡ hơn v3 nhờ train lâu hơn, không phải nhờ sửa đúng nguyên nhân gốc |
| v5 | 25/07/2026 | window=24, giảm drop_cosmos 0.85→0.5 | tương đương v4 | 16.75 | PPL không đổi (bác bỏ giả thuyết "mật độ cosmos" là nguyên nhân), nhưng cải thiện rõ khả năng quay lại cosmos theo thời gian |
| **v6-seq4096** | 26/07/2026 | **window=8 rebuild** (`w8_new`, 6 nguồn, giống kiến trúc v2) | 8,065 | 6.36 | Bản gần nhất với công thức v2, tốt nhất về khả năng quay lại `<agent>` (4 block liên tiếp, chưa từng thấy ở bản nào khác) |
| **v6-seq8192** | 26/07/2026 | window=8 rebuild, seq_length=8192 | 4,033 | **5.98** | Cân bằng nhất — PPL tốt nhất trong 3 bản v6, đủ 5/5 modal trong test full-chain |
| **v6-seq16384** | 26/07/2026 | window=8 rebuild, seq_length=16384 | 2,016 | 6.12 | Yếu nhất trong 3 bản v6 trên mọi trục — nghi do ít iteration nhất (cùng ngân sách token, seq dài hơn → ít bước train hơn) |

**Sự cố kỹ thuật đáng chú ý trong quá trình train v6**: double-submit vô tình tạo 5 job SLURM thay vì 3 (phát hiện + huỷ kịp thời, chưa checkpoint nào bị ảnh hưởng); symlink log (`current.log`) của 2/3 job trỏ nhầm về đúng job bị huỷ — nếu chỉ đọc log qua symlink sẽ tưởng nhầm training thất bại, log thật (theo đúng job ID từ `sacct`) vẫn đầy đủ và cho thấy cả 3 job COMPLETED sạch.

## 2.5. Phương pháp eval đầy đủ

Project dùng 3 loại eval bổ sung cho nhau (không loại nào thay được loại nào):

### (a) Sanity / Token atomicity
Kiểm tra 2 điều: (1) mọi token VLA có được tokenize thành đúng 1 token id hay không (atomicity), (2) model có hoàn thành đúng cấu trúc khi được mồi 1 phần dữ liệu thật hay không (ví dụ: mồi 3/17 khớp của 1 block agent thật, kỳ vọng model hoàn thành đúng 14 khớp còn lại với giá trị hợp lý). Script: `eval_vla_v2/v3/v6_sanity.py` — mỗi phiên bản window khác nhau cần record mồi thật tương ứng (window=8 khác window=24), tái dùng nhầm record sẽ test sai thứ model chưa từng học.

### (b) Temporal continuity (tính bền theo thời gian)
Câu hỏi: khi mồi model bằng 1 cửa sổ/chunk THẬT duy nhất rồi để tự sinh tiếp dài (sampled decoding), model có **tự quay lại đúng modal đó** theo thời gian hay không, và nội dung mới sinh có thật sự tiến triển (không đông cứng/lặp y hệt) hay không? Script: `eval_temporal_continuity.py`, đo bằng độ dịch chuyển giữa các cửa sổ pose liên tiếp (agent) và độ trùng lặp token vị trí (cosmos).

**Phát hiện chính**: v2 (window=8) bền vững hơn hẳn v3/v4/v5 (window=24) ở khả năng quay lại `<agent>`/`<cosmos>`. v6-seq4096 lần đầu tiên đạt **4 block agent liên tiếp không đông cứng** trong 1 lần sinh — kết quả tốt nhất từ trước đến nay, dù mới verify ở 1 seed (cần thêm seed để chắc chắn không phải ngẫu nhiên).

### (c) Full-chain text-to-media (kiểm tra thật sự "text → đa phương tiện")
Cho model 1 prompt HOÀN TOÀN MỚI (không có trong bất kỳ record training nào), để model tự sinh chuỗi modal, rồi **decode thật** mọi modal sinh ra thành file xem được: seed2→ảnh PNG, cosmos→video MP4, agent→pose JSON + ảnh khung xương, listen/speak→file âm thanh WAV. Script: `gen_full_chain_v3.py`.

**Phát hiện chính (và quan trọng nhất của toàn dự án)**: test với prompt "A person is running in a park" trên cả v2 và cả 3 bản v6 — **không bản nào sinh caption đúng chủ đề** (toàn bịa ra "riding a bike/slide/raft/airplane"). Đây là bằng chứng trực tiếp cho gap instruction-following đã nêu ở mục 6, Phần 1.

## 2.6. Hạ tầng & tooling

- **Cụm tính toán**: JUPITER, partition `booster` — node GH200, 4 GPU/node, 288 CPU core/node, account `reformo`.
- **Training framework**: Megatron-LM (qua wrapper `oellm-autoexp`), container Apptainer (chỉ bind `/e`, không thấy `/p`).
- **2 môi trường Python tách biệt** (không trộn): pipeline tokenize prototype (Seed2/Cosmos/AVC-LM) và pipeline pose 3D (HRNet/MotionBERT).
- **Vị trí dữ liệu chuẩn hoá** (sau đợt reorg 25/07/2026): `window8_legacy/` (bản cũ), `window24_current/` (thử nghiệm w24, nay không còn là "current" theo nghĩa thực tế), `w8_new/` (mix mới nhất dùng train v6) — tất cả dưới `/e/data1/datasets/playground/mmlaion/shared/nguyen38/`.
- **Repo chính**: `3d-human-pose/` (pipeline video+pose, chứa toàn bộ `data_prep/`, `tools/eval/`, `tools/upload/`), tách biệt với repo training `oellm-autoexp/`.

## 2.7. Hạn chế & vấn đề còn mở

1. **Instruction-following (ưu tiên #1)** — như đã nêu, gap lớn nhất, xuyên suốt mọi version, chưa được giải quyết bằng bất kỳ thay đổi pretraining nào đã thử.
2. **Modality persistence (đã cải thiện, chưa ổn định)** — v6 cải thiện rõ nhưng mới verify 1 seed cho phần lớn test; cần thêm seed để xác nhận không phải ngẫu nhiên.
3. **Chưa tách bạch được đóng góp riêng của window=8 vs các thay đổi đi kèm** — v2 và v6 khác nhau ở NHIỀU biến cùng lúc (mix nguồn, wrapper audio, oversample Harmony4D...), chưa có 1 ablation cô lập đúng 1 biến window size.
4. **Chưa có dữ liệu robot-action thật** — toàn bộ agent-token hiện tại đến từ ước lượng pose người qua video (không phải robot thật); DROID/GR00T-Sim là ứng viên nhưng chưa tích hợp.
5. **Model 1.7B có thể đang chạm trần năng lực** — nhồi quá nhiều modal vào 1 model tương đối nhỏ có chi phí thật (AnyGPT cũng tự báo cáo hiện tượng này) — chưa có bằng chứng rõ để tách bạch "do thiếu data" hay "do model quá nhỏ".
6. **Kỷ luật eval chưa theo kịp tốc độ mở rộng nguồn dữ liệu** — mỗi nguồn mới nên có eval tối thiểu đi kèm ngay, tránh lặp lại tình huống phải viết eval mới sau khi đã train xong (như đã xảy ra với w8_new).

## 2.8. Roadmap chi tiết

1. **VLA-Instruct** (ưu tiên cao nhất) — thiết kế: (a) dùng LLM sinh kịch bản hội thoại đa dạng dạng placeholder (giống công thức AnyInstruct: 100 meta-topic → hàng chục nghìn topic cụ thể); (b) **retrieval, không generate** — dùng embedding caption có sẵn để tìm clip/pose-window/audio thật khớp ngữ nghĩa trong dữ liệu đã tokenize, thay placeholder bằng token thật; (c) train như 1 **SFT stage riêng sau pretrain** (đúng recipe AnyGPT: pretrain → instruction-tune), không trộn chung vào mix pretrain 64-node hiện tại.
2. Chốt 1 seq_length cho v6 sau khi verify đa-seed (seq8192 đang có PPL tốt nhất + đủ 5 modal trong full-chain test).
3. Cân nhắc tích hợp DROID — cần thiết kế vocab action 7-DoF riêng, ưu tiên thấp hơn VLA-Instruct.
4. Publish model v6 (bản tốt nhất được chọn) lên HF, cập nhật `vla-1.7b-qwen3-v6` (chưa làm — hiện chỉ có checkpoint nội bộ).
5. Duy trì kỷ luật: mỗi nguồn dữ liệu mới cần có (a) verify license trước, (b) eval tối thiểu đi kèm, (c) không xoá bản dữ liệu cũ khi có bản mới (giữ lại để so sánh/rollback).

## 2.9. Glossary thuật ngữ

| Thuật ngữ | Giải thích |
|---|---|
| **VLA** | Vision-Language-Action — mô hình học đồng thời hình ảnh, ngôn ngữ, và hành động |
| **Token** | Đơn vị nhỏ nhất mà LLM xử lý — thường là 1 từ/mảnh từ; ở đây mở rộng để biểu diễn cả ảnh/video/âm thanh/hành động |
| **Tokenizer** | Bộ quy tắc biến dữ liệu gốc (chữ, ảnh, âm thanh...) thành chuỗi token |
| **Atomic token** | Token không bị tách nhỏ hơn nữa khi tokenize (quan trọng để model học đúng ý nghĩa 1 token = 1 đơn vị VLA) |
| **PPL (Perplexity)** | Chỉ số đo độ "bất ngờ" của model trước dữ liệu thật — càng thấp càng tốt, nghĩa là model dự đoán đúng dữ liệu tốt hơn |
| **Window** | Số khung hình gộp thành 1 "cửa sổ" thời gian khi tokenize video/pose (8 hoặc 24 khung hình) |
| **Modality drift** | Hiện tượng model "trôi" sang modal khác và không quay lại modal ban đầu dù dữ liệu thật có quay lại |
| **Instruction-following** | Khả năng model làm đúng theo yêu cầu/mô tả MỚI (chưa từng thấy), không chỉ tái tạo dữ liệu đã học |
| **Oversample** | Nhân bản 1 nguồn dữ liệu nhỏ lên N lần để tăng tỷ trọng của nó trong tổng dataset |
| **Sampled vs Greedy decoding** | Greedy = luôn chọn token có xác suất cao nhất (dễ bị lặp); Sampled = chọn ngẫu nhiên có trọng số theo xác suất (tự nhiên hơn, ít lặp hơn) |

## 2.10. Phụ lục — Tài nguyên & đường dẫn

**HuggingFace models** (org `EmpathicRobotics`):
- `vla-1.7b-qwen3-v2` — model public tốt nhất hiện tại
- `vla-1.7b-pab-spline-adaptive`, `vla-1.7b-pab-spline-25b-test` — model đời đầu (tokenizer bug)
- `tokenizer-vla-qwen3-v2` — tokenizer khuyến nghị hiện tại

**HuggingFace datasets** (org `EmpathicRobotics`):
- `FineVideo-Phase7-Flattened`, `FineVideo-Phase5-AgentTokens`, `FineVideo-Phase4-YOLOPose`, `FineVideo-Phase2-3DPose`, `FineVideo-Prototype-Tokenized`
- `harmony4d-flattened`, `omnivideo-100k-final`, `MV-Omni`, `synth-llava`, `emotional-roleplay-finetuning-dataset-flattened`

**Repo code**: `3d-human-pose/` (pipeline chính) — xem `CLAUDE.md` (hướng dẫn kỹ thuật), `datasets.md` (inventory dữ liệu chi tiết), `PROGRESS_VI.md`/`REPORT.md` (nhật ký phát triển đầy đủ theo thời gian).
