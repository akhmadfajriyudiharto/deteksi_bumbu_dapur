import sys
import cv2
import numpy as np
import base64
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QScrollArea, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class SpiceClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Bumbu Dapur: Akurasi Tinggi (Tanpa Counting)")
        self.setGeometry(100, 100, 1600, 900) 
        
        # Variabel penyimpanan gambar per tahap
        self.original_image = None
        self.step_1_blur = None         
        self.step_2_hsv = None          
        self.step_3_mask_red = None     
        self.step_4_mask_green = None   
        self.step_5_mask_white = None   
        self.step_6_global_mask = None  
        self.step_8_features = None     
        self.final_result_img = None    
        
        self.analysis_report = {
            "dominant_color": "Unknown",
            "aspect_ratio": "0.0",
            "prediction": "Belum diproses",
            "pixels": {"R": 0, "G": 0, "W": 0} # Inisialisasi agar tidak error
        }
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Panel Kontrol Kiri
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        self.btn_input = QPushButton("1. Input Gambar Bumbu")
        self.btn_input.clicked.connect(self.load_image)
        self.btn_input.setStyleSheet("padding: 10px; font-weight: bold;")
        control_layout.addWidget(self.btn_input)
        
        self.btn_process = QPushButton("2. Proses Klasifikasi")
        self.btn_process.clicked.connect(self.process_classification)
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet("padding: 10px; font-weight: bold; background-color: #d1ecf1;")
        control_layout.addWidget(self.btn_process)
        
        self.btn_report = QPushButton("3. Generate Laporan HTML")
        self.btn_report.clicked.connect(self.generate_report)
        self.btn_report.setEnabled(False)
        self.btn_report.setStyleSheet("padding: 10px; font-weight: bold;")
        control_layout.addWidget(self.btn_report)
        
        control_layout.addStretch(1)
        self.result_label = QLabel("### Hasil Identifikasi:")
        self.result_text = QLabel("Silakan Input Gambar.")
        self.result_text.setWordWrap(True)
        self.result_text.setStyleSheet("font-size: 18pt; font-weight: bold; color: #2c3e50;")
        control_layout.addWidget(self.result_label)
        control_layout.addWidget(self.result_text)
        control_layout.addStretch(3)
        main_layout.addWidget(control_widget, 1)

        # Area Scroll Gambar (Kanan)
        self.image_widgets = {}
        image_titles = [
            "1. Citra Asli (RGB)", 
            "2. Pre-processing (Blur)",
            "3. Ruang Warna HSV (Channel V)", 
            "4. Masking Merah (Pixel Count)",   
            "5. Masking Hijau (Pixel Count)",    
            "6. Masking Putih (Pixel Count)",   
            "7. Segmentasi Objek Utama", 
            "8. Analisis Bentuk (Box)",
            "9. Hasil Akhir"
        ]
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.image_grid_layout = QHBoxLayout(scroll_content)
        
        for title in image_titles:
            step_layout = QVBoxLayout()
            title_label = QLabel(f"### {title}")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
            
            image_label = QLabel("...")
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(250, 200)
            image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
            image_label.setStyleSheet("border: 2px dashed #aaa; background-color: #f9f9f9;")
            
            self.image_widgets[title] = image_label
            step_layout.addWidget(title_label)
            step_layout.addWidget(image_label)
            self.image_grid_layout.addLayout(step_layout, 1)
        
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, 4)

    # --- Utilities Gambar ---
    def convert_cv_to_qt(self, cv_img):
        if cv_img is None: return QPixmap()
        if len(cv_img.shape) == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = cv_img.shape
            bytes_per_line = 3 * w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = cv_img.shape
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def cv_to_base64(self, cv_img):
        if cv_img is None: return ""
        _, buffer = cv2.imencode('.jpg', cv_img)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"

    def update_display(self, title, img):
        if img is not None:
            pixmap = self.convert_cv_to_qt(img)
            label = self.image_widgets[title]
            scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Refresh visualisasi saat window di-resize
        if self.original_image is not None: self.update_display("1. Citra Asli (RGB)", self.original_image)
        if self.step_1_blur is not None: self.update_display("2. Pre-processing (Blur)", self.step_1_blur)
        if self.step_2_hsv is not None: self.update_display("3. Ruang Warna HSV (Channel V)", self.step_2_hsv)
        if self.step_3_mask_red is not None: self.update_display("4. Masking Merah (Pixel Count)", self.step_3_mask_red)
        if self.step_4_mask_green is not None: self.update_display("5. Masking Hijau (Pixel Count)", self.step_4_mask_green)
        if self.step_5_mask_white is not None: self.update_display("6. Masking Putih (Pixel Count)", self.step_5_mask_white)
        if self.step_6_global_mask is not None: self.update_display("7. Segmentasi Objek Utama", self.step_6_global_mask)
        if self.step_8_features is not None: self.update_display("8. Analisis Bentuk (Box)", self.step_8_features)
        if self.final_result_img is not None: self.update_display("9. Hasil Akhir", self.final_result_img)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                # Reset variable
                self.step_1_blur = self.step_2_hsv = self.step_3_mask_red = None
                self.step_4_mask_green = self.step_5_mask_white = self.step_6_global_mask = None
                self.step_8_features = self.final_result_img = None
                
                self.update_display("1. Citra Asli (RGB)", self.original_image)
                self.btn_process.setEnabled(True)
                self.result_text.setText("Gambar dimuat. Klik Proses.")
            else:
                QMessageBox.critical(self, "Error", "Gagal memuat gambar.")

    # --- INTI LOGIKA PEMROSESAN UTAMA ---
    def process_classification(self):
        if self.original_image is None: return
        
        # 1. Pre-processing
        # Menggunakan Median Blur untuk menghilangkan noise bintik ("salt and pepper")
        img_blur = cv2.medianBlur(self.original_image, 5)
        self.step_1_blur = img_blur.copy()
        self.update_display("2. Pre-processing (Blur)", self.step_1_blur)

        # 2. Convert to HSV
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        self.step_2_hsv = v.copy() 
        self.update_display("3. Ruang Warna HSV (Channel V)", self.step_2_hsv)

        # 3. Masking Warna (Definisi Range HSV yang Akurat)
        
        # MERAH: Diperluas sedikit range-nya agar cabai gelap terdeteksi
        lower_red1 = np.array([0, 40, 40])       # Saturation turun ke 40
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([155, 40, 40])     # Range 155-180
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # HIJAU: Range standar
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # PUTIH: DIPERKETAT
        # Saturation harus SANGAT rendah (0-40) agar tidak mendeteksi cabai merah pudar
        # Value harus TINGGI (140-255) agar tidak mendeteksi bayangan abu-abu
        lower_white = np.array([0, 0, 140])      
        upper_white = np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Simpan Masking untuk Display
        self.step_3_mask_red = mask_red
        self.step_4_mask_green = mask_green
        self.step_5_mask_white = mask_white
        self.update_display("4. Masking Merah (Pixel Count)", mask_red)
        self.update_display("5. Masking Hijau (Pixel Count)", mask_green)
        self.update_display("6. Masking Putih (Pixel Count)", mask_white)

        # 4. Global Masking & Pencarian Kontur
        # Gabungkan semua mask untuk menemukan letak objek di background apa pun
        combined_mask = mask_red | mask_green | mask_white
        
        # Morfologi: Closing (tutup lubang) lalu Opening (hilangkan noise luar)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        self.step_6_global_mask = combined_mask
        self.update_display("7. Segmentasi Objek Utama", combined_mask)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final_img = self.original_image.copy()
        
        if not contours:
            self.result_text.setText("Gagal: Tidak ada objek terdeteksi dalam range warna.")
            return

        # 5. FOKUS PADA 1 OBJEK TERBESAR SAJA (Abaikan noise kecil)
        c_main = max(contours, key=cv2.contourArea)
        area_main = cv2.contourArea(c_main)

        # --- FIX UTAMA: CEK APAKAH YANG TERDETEKSI ADALAH BACKGROUND? ---
        # Jika kontur menutupi > 90% gambar, berarti mask putih mendeteksi background.
        # Kita harus mengabaikan mask putih dan fokus mencari warna Merah/Hijau di dalamnya.
        h_img, w_img = self.original_image.shape[:2]
        img_area = h_img * w_img
        
        is_background_detected = False
        
        if area_main > 0.90 * img_area:
            print(">>> PERINGATAN: Background terdeteksi sebagai objek! Mencoba filter ulang...")
            is_background_detected = True
            
            # Coba cari ulang kontur HANYA dari mask Merah atau Hijau
            cnts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            new_c = None
            max_area_temp = 0
            
            # Cek jika ada objek merah yang signifikan
            if cnts_red:
                c_r = max(cnts_red, key=cv2.contourArea)
                if cv2.contourArea(c_r) > 1000: # Minimal area valid
                    new_c = c_r
                    max_area_temp = cv2.contourArea(c_r)
            
            # Cek jika ada objek hijau yang lebih besar
            if cnts_green:
                c_g = max(cnts_green, key=cv2.contourArea)
                if cv2.contourArea(c_g) > max_area_temp and cv2.contourArea(c_g) > 1000:
                    new_c = c_g
            
            if new_c is not None:
                c_main = new_c
                area_main = cv2.contourArea(c_main)
                # Visualisasi ulang bahwa kita mengganti target
                cv2.drawContours(final_img, [c_main], -1, (255, 0, 255), 3) # Warna ungu untuk 'Recovered Object'
                print(">>> SUKSES: Objek asli ditemukan di dalam background.")
            else:
                # Jika masih tidak ketemu (mungkin benar-benar bawang putih fullscreen)
                print(">>> GAGAL: Tetap menggunakan kontur background (beresiko salah).")

        if area_main < 1000:
            self.result_text.setText("Objek terlalu kecil untuk dianalisis.")
            return

        # Buat Mask khusus hanya untuk objek terbesar ini
        mask_object_only = np.zeros_like(combined_mask)
        cv2.drawContours(mask_object_only, [c_main], -1, 255, -1)

        # 6. LOGIKA VOTING PIXEL (DENGAN PRIORITAS WARNA)
        pixels_red = cv2.countNonZero(cv2.bitwise_and(mask_red, mask_object_only))
        pixels_green = cv2.countNonZero(cv2.bitwise_and(mask_green, mask_object_only))
        pixels_white = cv2.countNonZero(cv2.bitwise_and(mask_white, mask_object_only))
        
        # Jika tadi background terdeteksi, abaikan pixel putih yang bocor
        if is_background_detected:
            pixels_white = 0 
        
        total_pixels = pixels_red + pixels_green + pixels_white
        if total_pixels == 0: total_pixels = 1 

        # Hitung Persentase Warna
        pct_red = (pixels_red / total_pixels) * 100
        pct_green = (pixels_green / total_pixels) * 100
        
        print(f"PIXEL DATA -> Merah: {pixels_red} ({pct_red:.1f}%), Hijau: {pixels_green} ({pct_green:.1f}%), Putih: {pixels_white}")

        predicted_type = "Tidak Dikenali"
        dominant_color_name = "Campuran"
        aspect_ratio_val = 0.0
        solidity = 0.0
        circularity = 0.0
        
        # --- DECISION TREE BARU (PRIORITAS WARNA + CIRCULARITY) ---
        # Cabai merah sering punya pantulan putih (glare). 
        # Jadi, jika ada unsur merah minimal 10%, kita anggap itu BUKAN bawang putih.
        
        # 1. Cek HIJAU (Prioritas Tertinggi karena jarang glare)
        if pct_green > 10: 
            predicted_type = "CABAI HIJAU"
            dominant_color_name = "Hijau"
            # Hitung aspect ratio sekedar data
            rect = cv2.minAreaRect(c_main)
            w, h = rect[1]
            aspect_ratio_val = max(w, h) / min(w, h) if min(w,h) > 0 else 0

        # 2. Cek MERAH (Prioritas Kedua)
        # Jika merah > 40%, kemungkinan besar itu Cabai Merah atau Bawang Merah
        elif pct_red > 40:
            dominant_color_name = "Merah/Ungu"
            
            # Analisis Bentuk (Membedakan Cabai Merah vs Bawang Merah)
            # A. Aspect Ratio (Kelonjongan dari Rotated Rect)
            rect = cv2.minAreaRect(c_main)
            (cx, cy), (w, h), angle = rect
            dim_long = max(w, h)
            dim_short = min(w, h)
            aspect_ratio = dim_long / dim_short if dim_short > 0 else 0
            aspect_ratio_val = aspect_ratio
            
            # B. Solidity (Kepadatan Area Convex Hull)
            hull = cv2.convexHull(c_main)
            hull_area = cv2.contourArea(hull)
            solidity = area_main / hull_area if hull_area > 0 else 0

            # C. Circularity (Kebulatan - 4*pi*Area / Perimeter^2)
            # Cabai (panjang/kurus/melengkung) perimeter besar -> Circularity RENDAH (< 0.6)
            # Bawang (bulat) -> Circularity TINGGI (> 0.7)
            perimeter = cv2.arcLength(c_main, True)
            if perimeter == 0: perimeter = 1
            circularity = (4 * np.pi * area_main) / (perimeter ** 2)

            # Visualisasi Box & Hull
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(final_img, [box], 0, (255, 0, 0), 2) # Box Biru
            cv2.drawContours(final_img, [hull], 0, (0, 255, 255), 1) # Hull Kuning

            print(f"Shape -> Ratio: {aspect_ratio:.2f}, Solidity: {solidity:.2f}, Circ: {circularity:.2f}")

            # LOGIKA TRIPLE CHECK (Paling Akurat untuk Cabai Melengkung):
            # Cabai Merah jika memenuhi SALAH SATU syarat:
            # 1. Aspect Ratio > 1.3 (Agak lonjong sedikit saja dianggap cabai)
            # 2. ATAU Solidity < 0.8 (Tidak padat/Melengkung)
            # 3. ATAU Circularity < 0.6 (Objek kurus/memanjang pasti circularity rendah)
            
            if aspect_ratio > 1.6 or solidity < 0.8 or circularity < 0.4:
                predicted_type = "CABAI MERAH"
            else:
                predicted_type = "BAWANG MERAH"

        # 3. JIKA SANGAT SEDIKIT MERAH & HIJAU -> BARU BAWANG PUTIH
        else:
            predicted_type = "BAWANG PUTIH"
            dominant_color_name = "Putih"
            aspect_ratio_val = 1.0

        # 7. Visualisasi Hasil
        # Gambar kontur objek utama
        if not is_background_detected:
            cv2.drawContours(final_img, [c_main], -1, (0, 255, 0), 2)
        
        # Tampilkan Teks
        text_str = f"{predicted_type}"
        cv2.putText(final_img, text_str, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        self.step_8_features = final_img
        self.final_result_img = final_img
        self.update_display("8. Analisis Bentuk (Box)", self.step_8_features)
        self.update_display("9. Hasil Akhir", self.final_result_img)

        # Update UI Teks
        msg = f"Hasil Prediksi: {predicted_type}\nDominasi Warna: {dominant_color_name}\n"
        if dominant_color_name == "Merah/Ungu":
            msg += f"Rasio: {aspect_ratio_val:.2f}, Kepadatan: {solidity:.2f}, Kebulatan: {circularity:.2f}"
        
        if is_background_detected:
            msg += "\n(Background putih diabaikan)"
            
        self.result_text.setText(msg)
        self.btn_report.setEnabled(True)

        # Simpan Data untuk Report
        self.analysis_report = {
            "prediction": predicted_type,
            "dominant_color": dominant_color_name,
            "aspect_ratio": f"{aspect_ratio_val:.2f}",
            "pixels": {"R": pixels_red, "G": pixels_green, "W": pixels_white}
        }

    def generate_report(self):
        if self.final_result_img is None: return
        
        # Convert images to base64
        imgs = [
            self.cv_to_base64(self.original_image),
            self.cv_to_base64(self.step_1_blur),
            self.cv_to_base64(self.step_2_hsv),
            self.cv_to_base64(self.step_3_mask_red),
            self.cv_to_base64(self.step_4_mask_green),
            self.cv_to_base64(self.step_5_mask_white),
            self.cv_to_base64(self.step_6_global_mask),
            self.cv_to_base64(self.step_8_features),
            self.cv_to_base64(self.final_result_img)
        ]
        
        rep = self.analysis_report
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Laporan Klasifikasi Bumbu</title>
            <style>
                body {{ font-family: sans-serif; padding: 20px; }}
                h1 {{ color: #333; }}
                .box {{ border: 2px solid #333; padding: 15px; margin-bottom: 20px; background: #f0f8ff; }}
                .result {{ font-size: 24px; font-weight: bold; color: blue; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
                img {{ width: 100%; border: 1px solid #ccc; }}
                h4 {{ text-align: center; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>Laporan Analisis Citra Bumbu Dapur</h1>
            <p>Tanggal: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="box">
                <div>HASIL IDENTIFIKASI:</div>
                <div class="result">{rep['prediction']}</div>
            </div>
            
            <h3>Data Teknis:</h3>
            <table>
                <tr><th>Parameter</th><th>Nilai</th><th>Analisis</th></tr>
                <tr><td>Warna Dominan</td><td>{rep['dominant_color']}</td><td>Berdasarkan jumlah piksel terbanyak dalam kontur</td></tr>
                <tr><td>Rasio Bentuk (P/L)</td><td>{rep['aspect_ratio']}</td><td>Rasio > 2.0 indikasi Cabai (Lonjong), < 2.0 Bawang (Bulat)</td></tr>
                <tr><td>Jumlah Piksel Merah</td><td>{rep['pixels']['R']}</td><td>Area Masking Merah</td></tr>
                <tr><td>Jumlah Piksel Hijau</td><td>{rep['pixels']['G']}</td><td>Area Masking Hijau</td></tr>
                <tr><td>Jumlah Piksel Putih</td><td>{rep['pixels']['W']}</td><td>Area Masking Putih</td></tr>
            </table>

            <h3>Visualisasi Tahapan:</h3>
            <div class="grid">
                <div><img src="{imgs[0]}"><h4>1. Asli</h4></div>
                <div><img src="{imgs[1]}"><h4>2. Blur</h4></div>
                <div><img src="{imgs[2]}"><h4>3. HSV</h4></div>
                <div><img src="{imgs[3]}"><h4>4. Mask Red</h4></div>
                <div><img src="{imgs[4]}"><h4>5. Mask Green</h4></div>
                <div><img src="{imgs[5]}"><h4>6. Mask White</h4></div>
                <div><img src="{imgs[6]}"><h4>7. Segmentasi</h4></div>
                <div><img src="{imgs[7]}"><h4>8. Bentuk</h4></div>
                <div><img src="{imgs[8]}"><h4>9. Hasil</h4></div>
            </div>
        </body>
        </html>
        """
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Laporan", "Laporan_Bumbu.html", "HTML Files (*.html)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write(html)
            QMessageBox.information(self, "Sukses", f"Laporan disimpan di:\n{file_path}")

if __name__ == '__main__':
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    window = SpiceClassifierApp()
    window.show()
    sys.exit(app.exec_())