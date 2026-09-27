# Dog vs Cat Classification

จำแนกภาพสุนัขและแมวด้วย Transfer Learning (MobileNetV2) พัฒนาด้วย TensorFlow บน Python

---

## โครงสร้างโปรเจค

```
project/
├── data/
│   ├── train/
│   │   |── cat/
│   │   |── dog/
│   ├── val/
│   │   |── cat/
│   │   |── dog/
│   └── test/
│       |── cat/
│       |── dog/
├── outputs/
│   ├── models/
│   │   |── best_model.keras
│   |── plots/
│   |   |── sample_images.png
│   |   |── confusion_matrix.png
│   |   |── roc_curve.png
|   |   |── loss.png
|   |   |── wrong_predictions.png
│   └── prediction
        |── batch_result.png
        |── batch2_result.png
├── 1_explore_data.ipynb
├── 2_preprocess.ipynb
├── 3_train_model.ipynb
├── 4_evaluate.ipynb
└── 5_predict.ipynb
```

---

## Notebooks

โปรเจคจะแบ่งเป็น 5 ขั้นตอนชัดเจน แต่ละขั้นอยู่คนละไฟล์ เพื่อเรียนรู้เเต่ละขั้นตอน

**1_explore_data.ipynb** — ดูว่า dataset มีรูปกี่รูปแต่ละ split ตรวจสอบว่ารูปเป็น RGB ครบหรือเปล่า และ plot ตัวอย่างรูปจาก train set

**2_preprocess.ipynb** — เขียนฟังก์ชัน `load_and_preprocess()` ที่รับ path แล้ว resize เป็น 224×224 และ normalize ค่า pixel ให้อยู่ในช่วง 0–1 จากนั้นรันกับทุกรูปเพื่อตรวจว่าไม่มี error

**3_train_model.ipynb** — เทรนโมเดลด้วย MobileNetV2 (pretrained ImageNet) ใช้ ImageDataGenerator สำหรับ Data Augmentation และบันทึก best model ไว้ที่ `outputs/models/best_model.keras`

**4_evaluate.ipynb** — โหลดโมเดลที่เทรนแล้วมาทดสอบกับ test set แสดง Classification Report, Confusion Matrix และ ROC Curve

**5_predict.ipynb** — โหลดโมเดลมาใช้ทาย ทั้งแบบรูปเดียว (`predict_image`) และแบบหลายรูปพร้อมกัน (`predict_batch`)

---

## โมเดล

ใช้ MobileNetV2 ที่ pretrained บน ImageNet เป็น base model แล้ว freeze ไว้ ต่อด้วย layer ของตัวเอง

MobileNetV2 (frozen, pretrained ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(1, Sigmoid) → cat / dog

เลือก MobileNetV2 เพราะ เป็นโมเดลที่มีขนาดเล็กและมีประสิทธิภาพในการประมวลผลสูง ทำงานได้เร็วบน CPU โดยไม่จำเป็นต้องพึ่งพา GPU ทำให้สะดวกต่อการพัฒนา ทดสอบ และฝึกโมเดลบน VS Code แล้วก็ลดปัญหการ timeout ในระหว่างการทำงาน

---

## Training Config

| Parameter     | ค่า |

| Image size    | 224 × 224 |
| Batch size    | 32        |
| Epochs        | 10        |
| Learning rate | 0.001     |
| Optimizer     | Adam      |
| Loss          | Binary Crossentropy |

**Data Augmentation ที่ใช้**

- Horizontal flip
- Rotation ±15°
- Zoom 20%
- Width/Height shift 10%

---

## ผลลัพธ์

Test Accuracy   98.17%
AUC (ROC)       0.9993
Cat — ทายถูก / ผิด    292 / 8
Dog — ทายถูก / ผิด    299 / 1

---

## วิธีรัน

รัน Notebook ตามลำดับจากบนลงล่าง

```
1_explore_data.ipynb
2_preprocess.ipynb
3_train_model.ipynb
4_evaluate.ipynb
5_predict.ipynb
```

ทดสอบกับรูปใหม่ได้ที่ `5_predict.ipynb` เปลี่ยน `img_path` เป็น path รูปที่ต้องการ

---

## Dependencies

```
tensorflow
numpy
matplotlib
pillow
scikit-learn
seaborn
```

---

