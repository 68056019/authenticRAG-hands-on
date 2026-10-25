# Authentic RAG v2

Authentic RAG v2 เป็นระบบถาม–ตอบจากเอกสาร โดยค้นหาข้อมูลที่เกี่ยวข้องก่อน แล้วจึงส่งเฉพาะหลักฐานที่ค้นพบให้โมเดลสร้างคำตอบ

v2 ปรับปรุงวิธีแบ่งเอกสาร การสร้าง embedding การค้นหา การรวมคะแนน และการจัดอันดับใหม่ ส่วนโมเดลที่สร้างคำตอบสุดท้ายยังเป็น `qwen/qwen-2.5-72b-instruct` เหมือนระบบเดิม

## ระบบทำงานอย่างไร

ระบบแบ่งเป็น 3 ส่วนหลัก และทำงานต่อเนื่องกันดังนี้

```text
เอกสาร Markdown
        ↓
1. Ingest — เตรียมเอกสารและสร้างดัชนีค้นหา
        ↓
Vector Index + Lexical Index
        ↓
คำถามของผู้ใช้
        ↓
2. Retrieve — ค้นหาและจัดอันดับหลักฐาน
        ↓
หลักฐานที่เกี่ยวข้องที่สุด
        ↓
3. Answer — ใช้หลักฐานสร้างคำตอบ
        ↓
คำตอบสุดท้ายพร้อมข้อมูลอ้างอิง
```

## ส่วนที่ 1: เตรียมเอกสาร (Ingest)

ส่วนนี้ทำงานก่อนเริ่มถามคำถาม มีหน้าที่เปลี่ยนไฟล์ Markdown ให้เป็นข้อมูลที่ค้นหาได้อย่างมีประสิทธิภาพ

### Input

- โฟลเดอร์ที่เก็บไฟล์ Markdown
- ตัวเลือกว่าจะสร้าง index ใหม่หรือใช้ของเดิม
- ตัวเลือกว่าจะใช้ LLM เพิ่มบริบทให้แต่ละ chunk หรือไม่

### ขั้นตอนการทำงาน

1. ระบบอ่านไฟล์ Markdown ทีละไฟล์ (`ไฟล์ Markdown → ข้อความ`)
2. แยกเนื้อหาตามโครงสร้างหัวข้อ เพื่อไม่ให้เนื้อหาคนละเรื่องปะปนกัน (`ข้อความ → ตรวจ Markdown headings → sections พร้อม breadcrumb`)
3. แบ่งเนื้อหาเป็น `chunk` พร้อมเก็บข้อความบางส่วนซ้ำระหว่างชิ้น เพื่อไม่ให้ข้อมูลตรงรอยต่อหายไป (`section → ย่อหน้า → chunks พร้อม overlap`)
4. สร้างรหัสประจำ chunk จากชื่อไฟล์ ลำดับ และเนื้อหา ทำให้อ้างอิง chunk เดิมได้อย่างสม่ำเสมอ (`ชื่อไฟล์ + ลำดับ + เนื้อหา → SHA-1 → stable chunk ID`)
5. ถ้าเปิด contextualization ระบบจะให้ LLM เขียนคำอธิบายสั้น ๆ ว่า chunk กล่าวถึงอะไรและอยู่ส่วนใดของเอกสาร (`document excerpt + heading + chunk → prompt → tokenizer → token IDs → LLM → output tokens → context text`)
6. รวมชื่อหัวข้อ context และเนื้อหา แล้วส่งเข้า BGE-M3 เพื่อสร้าง embedding vector (`heading + context + content → retrieval instruction → BGE-M3 tokenizer → token IDs → embedding model → normalized vector`)
7. บันทึกข้อมูลลงสอง index (`normalized vector + metadata → Vector Index` และ `chunk text → character n-gram tokenizer → lexical tokens → Lexical Index`)

token IDs ในขั้นตอนที่ 5 และ 6 เป็นข้อมูลภายในของโมเดล โค้ดจึงส่งข้อความเข้าไปและรับ context หรือ vector กลับมาโดยไม่ต้องจัดเก็บ token IDs เอง

### Output

- Vector Index ซึ่งเก็บ embedding และข้อมูลของแต่ละ chunk
- Lexical Index ซึ่งเก็บข้อความสำหรับค้นหาแบบตรงคำ
- จำนวน chunks ที่บันทึกสำเร็จ

## ส่วนที่ 2: ค้นหาและจัดอันดับหลักฐาน (Retrieve)

เมื่อผู้ใช้ส่งคำถามเข้ามา ระบบจะค้นหาหลักฐานจาก index ทั้งสองแบบ แล้วรวมผลลัพธ์ก่อนจัดอันดับใหม่

### Input

- คำถามของผู้ใช้ (`question`)
- จำนวนหลักฐานสุดท้ายที่ต้องการ (`top_k`)
- จำนวน candidate ที่ต้องการดึงจากแต่ละวิธี (`candidates`)

### ขั้นตอนการทำงาน

1. **Dense search:** เปลี่ยนคำถามเป็น query vector แล้วค้นหา chunks ที่มีความหมายใกล้เคียง วิธีนี้ช่วยค้นหาคำพ้องหรือข้อความที่ใช้คนละคำแต่มีความหมายคล้ายกัน (`คำถาม → BGE-M3 tokenizer → token IDs → embedding model → query vector → Vector Index → dense hits`)
2. **Lexical search:** แยกคำถามเป็น character n-grams แล้วค้นหาข้อความที่ตรงกัน วิธีนี้ช่วยรักษาชื่อเฉพาะ ชื่อโรค ตัวเลข และคำสำคัญ (`คำถาม → character n-gram tokenizer → lexical tokens → Lexical Index → lexical hits`)
3. **Normalize scores:** ปรับคะแนนของแต่ละวิธีให้อยู่ในช่วง 0–1 (`raw Dense/Lexical scores → min-max normalization → normalized scores`)
4. **Weighted RRF:** รวมลำดับจาก Dense และ Lexical search โดยให้น้ำหนัก Dense 0.58 และ Lexical 0.42 (`dense hits + lexical hits → weighted reciprocal ranks → retrieval scores → candidate chunks`)
5. **Cross-encoder reranking:** ให้ reranker อ่านคำถามคู่กับ candidate แต่ละ chunk แล้วประเมินความเกี่ยวข้องโดยตรง (`question/chunk pairs → Cross-encoder tokenizer → token IDs → reranker → normalized relevance scores`)
6. **Final scoring:** รวมคะแนน reranker 72% กับคะแนน retrieval 28% (`reranker score × 0.72 + retrieval score × 0.28 → final score`)
7. **Diversity selection:** ลดการเลือก chunks ที่มีเนื้อหาซ้ำกันมากเกินไป เพื่อให้หลักฐานครอบคลุมหลายประเด็น (`candidate text → BGE-M3 embeddings → cosine redundancy penalty → เลือกหลักฐาน top-k`)

### Output

รายการหลักฐานที่เกี่ยวข้องที่สุด แต่ละรายการมีเนื้อหา chunk ชื่อไฟล์ ชื่อหัวข้อ ช่องทางที่ค้นพบ และคะแนนที่ใช้จัดอันดับ

## ส่วนที่ 3: สร้างคำตอบ (Answer)

ส่วนนี้นำหลักฐานจาก Retrieve มาสร้างคำตอบที่อ่านเข้าใจง่าย โดยไม่ได้เปลี่ยนโมเดลสร้างคำตอบจากระบบเดิม

### Input

- คำถามของผู้ใช้
- จำนวนหลักฐานที่ต้องการใช้
- หลักฐานที่ส่วน Retrieve เลือกมา

### ขั้นตอนการทำงาน

1. เรียก Retrieve เพื่อขอหลักฐานที่เกี่ยวข้องที่สุด (`คำถาม + top_k → Retrieve → evidence chunks`)
2. รวมชื่อไฟล์ ชื่อหัวข้อ context และเนื้อหาของหลักฐานเป็น prompt (`evidence chunks → จัดรูปแบบ context → user prompt`)
3. กำหนดให้ LLM ตอบจากหลักฐานที่ให้เท่านั้น และแจ้งเมื่อข้อมูลไม่เพียงพอ (`system instruction + user prompt → messages`)
4. ส่ง messages ผ่าน OpenRouter ไปยังโมเดลเดิม (`messages → tokenizer ภายใน LLM → input token IDs → qwen/qwen-2.5-72b-instruct → output tokens`)
5. ถอด output tokens กลับเป็นข้อความคำตอบ (`output tokens → decoder → answer text`)
6. ส่งคืนคำตอบพร้อมหลักฐาน เพื่อให้ตรวจสอบที่มาได้ (`question + answer text + evidence + model names → Dictionary/JSON`)

### Output

ผลลัพธ์เป็น Dictionary/JSON ซึ่งประกอบด้วย:

- `question`: คำถามต้นฉบับ
- `answer`: คำตอบสุดท้าย
- `results`: หลักฐานที่ใช้ประกอบคำตอบ
- `models`: รายชื่อ embedding model, reranker และ final-answer model

## หน้าที่ของฟังก์ชันต่าง ๆ

### การแบ่งเอกสาร

- `Chunk.retrieval_text` รวม heading, context และ content เป็นข้อความสำหรับค้นหา
- `MarkdownStructureChunker.__init__` รับและตรวจสอบค่าขนาด chunk กับ overlap
- `split_file` อ่านไฟล์ แบ่งเนื้อหา และสร้างรายการ `Chunk`
- `_sections` แยก Markdown ตามหัวข้อและสร้าง breadcrumb
- `flush` นำบรรทัดที่สะสมไว้มาสร้าง section
- `_window` รวมย่อหน้าเป็น chunk ตามขนาดเป้าหมาย
- `_split_oversized` แบ่งย่อหน้าที่ยาวเกินกำหนด
- `_tail` เลือกข้อความท้าย chunk สำหรับใช้เป็น overlap

### การเตรียมระบบและ index

- `AuthenticRAGv2.__init__` เชื่อมต่อ OpenRouter/OpenSearch และโหลด embedding model กับ reranker
- `create_indices` สร้าง Vector Index และ Lexical Index
- `contextualize` ใช้ LLM สร้าง context ให้ chunk
- `ingest` ควบคุมกระบวนการเตรียมเอกสารทั้งหมด
- `rebuild_lexical_from_vector` สร้าง Lexical Index ใหม่จากข้อมูลใน Vector Index
- `_embedding_text` เติม instruction ก่อนส่ง passage เข้า BGE-M3

### การค้นหาและให้คะแนน

- `retrieve` ควบคุมกระบวนการค้นหาและจัดอันดับทั้งหมด
- `_weighted_rrf` รวมผล Dense และ Lexical search
- `_score_map` สร้างตารางคะแนน normalized ของเอกสารแต่ละรายการ
- `_minmax` ปรับคะแนนให้อยู่ในช่วง 0–1
- `_diversify` เลือกหลักฐานที่เกี่ยวข้องและไม่ซ้ำกันมากเกินไป

### การสร้างคำตอบและรับคำสั่ง

- `answer` ค้นหลักฐาน ประกอบ prompt และเรียก LLM สร้างคำตอบ
- `build_parser` สร้างคำสั่งและตัวเลือกของ CLI
- `main` อ่านคำสั่ง เลือก workflow และแสดงหรือบันทึกผลลัพธ์

รายละเอียด Input, Data flow และ Output ของทุกฟังก์ชันอยู่ใน docstring ภายใน `authenticRAG_v2.py` เพื่อให้ดูพร้อมกับโค้ดที่ทำงานจริงได้

## วิธีรัน

### 1. ตั้งค่า API key

```powershell
$env:OPENROUTER_API_KEY = "ใส่-key-ที่นี่"
```

อย่าบันทึก API key จริงลงใน source code หรือไฟล์ Markdown

### 2. นำเอกสารเข้า index

```powershell
python authenticRAG_v2.py ingest --corpus ./corpus_input --recreate
```

หากไม่ต้องการเรียก LLM เพื่อเพิ่ม context ให้ใช้:

```powershell
python authenticRAG_v2.py ingest --corpus ./corpus_input --recreate --no-contextualize
```

### 3. ถามคำถามหนึ่งข้อ

```powershell
python authenticRAG_v2.py search --question "คำถามที่ต้องการค้นหา" --top-k 5
```

### 4. ประมวลผลคำถามหลายข้อ

ไฟล์คำถามต้องมีหนึ่งคำถามต่อหนึ่งบรรทัด:

```powershell
python authenticRAG_v2.py batch --questions questions.txt --output results_v2.json
```

### 5. สร้าง Lexical Index ใหม่

ใช้เมื่อแก้การตั้งค่า lexical analyzer และต้องการสร้างเฉพาะ Lexical Index ใหม่ โดยไม่ต้องสร้าง embedding ซ้ำ:

```powershell
python authenticRAG_v2.py rebuild-lexical
```
