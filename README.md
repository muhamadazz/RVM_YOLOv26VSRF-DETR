# Evaluasi YOLO26 dan RF-DETR untuk Deteksi Botol Plastik

Repository ini berisi materi penelitian yang membandingkan model YOLO26 dan RF-DETR untuk mendeteksi botol plastik pada konteks *Reverse Vending Machine* (RVM). Isi repository mencakup notebook pelatihan dan evaluasi, modul inferensi, serta contoh catatan hasil deteksi dan waktu inferensi.

## Isi Repository

| Direktori/berkas | Keterangan |
| --- | --- |
| `notebooks/codetraining_modelyolov26_skripsi.ipynb` | Notebook eksperimen dan pelatihan YOLO26. |
| `notebooks/codetraining_modelRF-DETR_skripsi.ipynb` | Notebook eksperimen dan pelatihan RF-DETR Nano. |
| `RVM_inference/yolo_model.py` | Fungsi inferensi gambar menggunakan YOLO26. |
| `RVM_inference/rfdetr_model.py` | Fungsi inferensi gambar menggunakan RF-DETR Nano. |
| `RVM_inference/inference_logger.py` | Pencatatan waktu inferensi, FPS, jumlah deteksi, dan statistik. |
| `RVM_inference/log_viewer.py` | Utilitas untuk melihat perbandingan log dan mengekspor log ke CSV. |
| `RVM_testRecords/` | Contoh data hasil deteksi dan catatan waktu inferensi dalam format JSON. |

## Dataset

Dataset yang dirujuk dalam repository tersedia di [Roboflow Universe](https://universe.roboflow.com/uascv/dataset_yolov26_efficientdet). Notebook mungkin menggunakan versi atau konfigurasi dataset yang berbeda. Untuk perbandingan yang adil, pastikan kedua model dievaluasi pada split data dan versi dataset yang sama.

Pengunduhan dataset melalui Roboflow memerlukan API key. Gunakan API key milik sendiri dan simpan melalui environment atau secrets notebook; jangan membagikan atau mengunggah API key ke repository.

## Persiapan

Notebook dirancang untuk dijalankan di lingkungan Jupyter/Google Colab. GPU NVIDIA direkomendasikan, terutama untuk pelatihan; kebutuhan GPU dan performa inferensi dapat berbeda sesuai perangkat.

1. Clone repository dan buka salah satu notebook dari folder `notebooks/`.
2. Siapkan lingkungan Python/Jupyter. Instal dependensi sesuai notebook yang dipakai:

   ```bash
   pip install ultralytics
   ```

   Untuk notebook RF-DETR:

   ```bash
   pip install "rfdetr[train,loggers]" supervision roboflow
   ```

3. Siapkan dataset dan kredensial Roboflow bila menjalankan sel yang mengunduh dataset.
4. Sesuaikan path dataset, checkpoint, dan penyimpanan hasil dengan lingkungan yang digunakan. Beberapa sel notebook menggunakan path khusus Google Colab seperti `/content/...`.

Notebook berisi beberapa bagian eksperimen dan contoh tambahan. Jalankan sel secara berurutan pada bagian yang relevan, dan periksa kembali konfigurasi/path sebelum memulai pelatihan.

## Checkpoint Model untuk Inferensi

Berkas bobot model tidak disertakan dalam repository. Modul inferensi mengharapkan checkpoint berikut tersedia relatif terhadap direktori kerja:

```text
plasticBottleDetection/
├── best_plasticbottle_yolov26.pt
└── checkpoint_best_total.pth
```

Letakkan checkpoint hasil pelatihan pada path tersebut, atau ubah path di `RVM_inference/yolo_model.py` dan `RVM_inference/rfdetr_model.py` agar sesuai dengan lokasi berkas bobot Anda. Modul YOLO memuat checkpoint saat modul diimpor; modul RF-DETR memuatnya saat pertama kali inferensi dipanggil.

## Contoh Pemakaian Modul Inferensi

Jalankan dari direktori utama repository setelah dependensi dan checkpoint yang diperlukan tersedia:

```python
from RVM_inference.yolo_model import detect_image

predictions, metrics = detect_image(
    image_path="path/to/image.jpg",
    image_name="image.jpg",
)

print(predictions)
print(metrics)
```

Untuk RF-DETR:

```python
from RVM_inference.rfdetr_model import detect_image

predictions, metrics = detect_image(
    image_path="path/to/image.jpg",
    image_name="image.jpg",
)
```

Kedua fungsi mengembalikan pasangan `(predictions, metrics)`. Prediksi mencakup koordinat dan ukuran bounding box, confidence, kelas, serta label perkiraan ukuran botol (`330ml`, `600ml`, atau `1500ml`). Label ukuran dihitung dari luas bounding box dan merupakan heuristik, bukan pengukuran kapasitas botol secara fisik. Metrik inferensi mencakup waktu, FPS, dan jumlah deteksi.

Log inferensi baru disimpan otomatis sebagai JSON di direktori `inference_logs/` pada root repository. Untuk membaca statistik log:

```python
from RVM_inference.inference_logger import InferenceLogger

print(InferenceLogger.get_statistics())
```

## Hasil Eksperimen

Contoh catatan hasil tersedia di `RVM_testRecords/`. Berkas `RVM_detectionTarget_*.json` dan `RVM_detectionNonTarget_*.json` berisi sampel prediksi, sedangkan `*_TimingLogs.json` berisi catatan waktu. Angka pada berkas tersebut berasal dari eksperimen tertentu; waktu dan FPS dapat berubah bergantung pada perangkat, konfigurasi, ukuran gambar, dan kondisi runtime.

## Catatan

- Pastikan versi dataset, split evaluasi, ukuran input, dan kondisi perangkat dicatat saat membandingkan model.
- Notebook dapat mengunduh dependensi dan dataset dari layanan eksternal. Tinjau sel notebook sebelum menjalankannya dan gunakan kredensial pribadi melalui mekanisme secrets.
- Sesuaikan path checkpoint dan path dataset sebelum menjalankan modul atau notebook di luar lingkungan asalnya.
