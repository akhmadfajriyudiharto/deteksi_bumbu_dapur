# 🌶️ Aplikasi Klasifikasi Bumbu Dapur

> **Proyek Akhir Mata Kuliah Multimedia Data Processing**
>
> **Disusun Oleh:**
> * **Nama:** Akhmad Fajri Yudiharto
> * **NIM:** 225210722
>
> **Institut Sains dan Teknologi Terpadu Surabaya**

---

## 📖 Deskripsi Proyek

Aplikasi desktop berbasis Python ini dirancang untuk mengidentifikasi dan mengklasifikasikan jenis bumbu dapur utama (**Cabai Merah, Cabai Hijau, Bawang Merah, dan Bawang Putih**) menggunakan teknik pengolahan citra digital (*Computer Vision*).

Aplikasi ini menerapkan pendekatan *Classic Computer Vision* (Non-Deep Learning) yang mengandalkan analisis ruang warna (HSV), operasi morfologi, dan analisis fitur bentuk geometris untuk mencapai akurasi tinggi dengan komputasi yang efisien.

## ✨ Fitur Utama

* **GUI Interaktif:** Antarmuka pengguna yang dibangun menggunakan **PyQt5** yang intuitif.
* **Visualisasi Pipeline (9 Tahap):** Menampilkan proses pengolahan citra langkah demi langkah, mulai dari citra asli, *blurring*, *masking* warna, segmentasi, hingga hasil akhir.
* **Analisis Warna & Bentuk:** Menggabungkan deteksi warna dominan dan analisis fitur bentuk (*Aspect Ratio, Solidity, Circularity*) untuk membedakan objek dengan warna serupa (misalnya: membedakan Cabai Merah yang melengkung dengan Bawang Merah yang bulat).
* **Penanganan Background:** Dilengkapi logika untuk mendeteksi jika *background* teridentifikasi sebagai objek, dan secara otomatis melakukan pencarian ulang objek asli di dalamnya.
* **Laporan Otomatis:** Fitur untuk mencetak laporan analisis dalam format **HTML** beserta lampiran gambar hasil proses.

## 🛠️ Requirements (Prasyarat)

Pastikan Python 3.x sudah terinstall. Library yang digunakan dalam proyek ini adalah:

* **Python 3.x**
* **OpenCV** (`cv2`) - Pemrosesan citra digital.
* **NumPy** - Operasi matriks dan numerik.
* **PyQt5** - Framework antarmuka grafis (GUI).

### Instalasi Library
Jalankan perintah berikut pada terminal:

```bash
pip install opencv-python numpy PyQt5