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
        self.setWindowTitle("Klasifikasi Bumbu Dapur: Laporan Analisis Lengkap")
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
        
        # Inisialisasi dictionary laporan
        self.analysis_report = {
            "prediction": "Belum diproses",
            "dominant_color": "Unknown",
            "aspect_ratio": "0.0",
            "solidity": "0.0",
            "circularity": "0.0",
            "metrics": {"pct_red": 0, "pct_green": 0, "pct_white": 0},
            "pixels": {"R": 0, "G": 0, "W": 0},
            "logic_path": "Tidak ada data"
        }
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # --- Panel Kontrol Kiri ---
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
        
        self.btn_report = QPushButton("3. Generate Laporan Detail (HTML)")
        self.btn_report.clicked.connect(self.generate_report)
        self.btn_report.setEnabled(False)
        self.btn_report.setStyleSheet("padding: 10px; font-weight: bold; background-color: #d4edda;")
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

        # --- Area Scroll Gambar (Kanan) ---
        self.image_widgets = {}
        image_titles = [
            "1. Citra Asli (RGB)", 
            "2. Pre-processing (Blur)",
            "3. Ruang Warna HSV (Channel V)", 
            "4. Masking Merah (Pixel Count)",   
            "5. Masking Hijau (Pixel Count)",    
            "6. Masking Putih (Pixel Count)",   
            "7. Segmentasi Objek Utama", 
            "8. Analisis Bentuk (Box & Hull)",
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
        if self.step_8_features is not None: self.update_display("8. Analisis Bentuk (Box & Hull)", self.step_8_features)
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

    # --- LOGIKA PEMROSESAN ---
    def process_classification(self):
        if self.original_image is None: return
        
        # 1. Pre-processing (Median Blur)
        img_blur = cv2.medianBlur(self.original_image, 5)
        self.step_1_blur = img_blur.copy()
        self.update_display("2. Pre-processing (Blur)", self.step_1_blur)

        # 2. HSV Conversion
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        self.step_2_hsv = v.copy() 
        self.update_display("3. Ruang Warna HSV (Channel V)", self.step_2_hsv)

        # 3. Masking Warna
        # MERAH
        lower_red1, upper_red1 = np.array([0, 40, 40]), np.array([15, 255, 255])
        lower_red2, upper_red2 = np.array([155, 40, 40]), np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # HIJAU
        lower_green, upper_green = np.array([35, 40, 40]), np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # PUTIH
        lower_white, upper_white = np.array([0, 0, 140]), np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        self.step_3_mask_red = mask_red
        self.step_4_mask_green = mask_green
        self.step_5_mask_white = mask_white
        self.update_display("4. Masking Merah (Pixel Count)", mask_red)
        self.update_display("5. Masking Hijau (Pixel Count)", mask_green)
        self.update_display("6. Masking Putih (Pixel Count)", mask_white)

        # 4. Global Masking & Contours
        combined_mask = mask_red | mask_green | mask_white
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        self.step_6_global_mask = combined_mask
        self.update_display("7. Segmentasi Objek Utama", combined_mask)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_img = self.original_image.copy()
        
        if not contours:
            self.result_text.setText("Gagal: Tidak ada objek terdeteksi.")
            return

        # 5. Cek Objek Utama vs Background
        c_main = max(contours, key=cv2.contourArea)
        area_main = cv2.contourArea(c_main)
        h_img, w_img = self.original_image.shape[:2]
        
        is_bg_detected = False
        if area_main > 0.90 * (h_img * w_img):
            is_bg_detected = True
            cnts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            new_c = None
            max_area_temp = 0
            if cnts_red:
                c_r = max(cnts_red, key=cv2.contourArea)
                if cv2.contourArea(c_r) > 1000:
                    new_c, max_area_temp = c_r, cv2.contourArea(c_r)
            if cnts_green:
                c_g = max(cnts_green, key=cv2.contourArea)
                if cv2.contourArea(c_g) > max_area_temp and cv2.contourArea(c_g) > 1000:
                    new_c = c_g
            
            if new_c is not None:
                c_main = new_c
                area_main = cv2.contourArea(c_main)

        # 6. Hitung Pixel dalam Mask Objek
        mask_object_only = np.zeros_like(combined_mask)
        cv2.drawContours(mask_object_only, [c_main], -1, 255, -1)

        pixels_red = cv2.countNonZero(cv2.bitwise_and(mask_red, mask_object_only))
        pixels_green = cv2.countNonZero(cv2.bitwise_and(mask_green, mask_object_only))
        pixels_white = 0 if is_bg_detected else cv2.countNonZero(cv2.bitwise_and(mask_white, mask_object_only))
        
        total_pixels = pixels_red + pixels_green + pixels_white
        if total_pixels == 0: total_pixels = 1 

        pct_red = (pixels_red / total_pixels) * 100
        pct_green = (pixels_green / total_pixels) * 100
        pct_white = (pixels_white / total_pixels) * 100

        # 7. Klasifikasi
        predicted_type = "Tidak Dikenali"
        dominant_color_name = "Campuran"
        logic_explanation = ""
        
        # Hitung Shape Features (Default)
        rect = cv2.minAreaRect(c_main)
        (cx, cy), (w, h), angle = rect
        aspect_ratio_val = max(w, h) / min(w, h) if min(w,h) > 0 else 0
        
        hull = cv2.convexHull(c_main)
        hull_area = cv2.contourArea(hull)
        solidity = area_main / hull_area if hull_area > 0 else 0
        
        perimeter = cv2.arcLength(c_main, True)
        if perimeter == 0: perimeter = 1
        circularity = (4 * np.pi * area_main) / (perimeter ** 2)

        # VISUALISASI FITUR (Box & Hull)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(final_img, [box], 0, (255, 0, 0), 2) # Box Biru
        cv2.drawContours(final_img, [hull], 0, (0, 255, 255), 1) # Hull Kuning

        # --- DECISION TREE LOGIC ---
        if pct_green > 10: 
            predicted_type = "CABAI HIJAU"
            dominant_color_name = "Hijau"
            logic_explanation = "Terdeteksi piksel HIJAU signifikan (>10%)."
        
        elif pct_red > 40:
            dominant_color_name = "Merah/Ungu"
            # Cek Bentuk
            if aspect_ratio_val > 1.6 or solidity < 0.8 or circularity < 0.4:
                predicted_type = "CABAI MERAH"
                logic_explanation = "Warna MERAH dominan. Bentuk MEMANJANG (Ratio tinggi) atau MELENGKUNG (Solidity rendah)."
            else:
                predicted_type = "BAWANG MERAH"
                logic_explanation = "Warna MERAH dominan. Bentuk BULAT/PADAT (Ratio rendah, Solidity tinggi)."
        
        else:
            predicted_type = "BAWANG PUTIH"
            dominant_color_name = "Putih"
            logic_explanation = "Minim warna Merah/Hijau. Didominasi warna PUTIH."

        # Update Visual Akhir
        if not is_bg_detected:
            cv2.drawContours(final_img, [c_main], -1, (0, 255, 0), 2)
        
        cv2.putText(final_img, predicted_type, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        self.step_8_features = final_img
        self.final_result_img = final_img
        self.update_display("8. Analisis Bentuk (Box & Hull)", self.step_8_features)
        self.update_display("9. Hasil Akhir", self.final_result_img)

        # Update UI & Save Report Data
        self.result_text.setText(f"Hasil: {predicted_type}\n({logic_explanation})")
        self.btn_report.setEnabled(True)

        self.analysis_report = {
            "prediction": predicted_type,
            "dominant_color": dominant_color_name,
            "aspect_ratio": f"{aspect_ratio_val:.2f}",
            "solidity": f"{solidity:.2f}",
            "circularity": f"{circularity:.2f}",
            "metrics": {
                "pct_red": f"{pct_red:.1f}", 
                "pct_green": f"{pct_green:.1f}", 
                "pct_white": f"{pct_white:.1f}"
            },
            "pixels": {"R": pixels_red, "G": pixels_green, "W": pixels_white},
            "logic_path": logic_explanation
        }

    def generate_report(self):
        if self.final_result_img is None: return
        
        # Helper images
        imgs = [
            self.cv_to_base64(self.original_image),      # 0
            self.cv_to_base64(self.step_1_blur),         # 1
            self.cv_to_base64(self.step_2_hsv),          # 2
            self.cv_to_base64(self.step_3_mask_red),     # 3
            self.cv_to_base64(self.step_4_mask_green),   # 4
            self.cv_to_base64(self.step_5_mask_white),   # 5
            self.cv_to_base64(self.step_6_global_mask),  # 6
            self.cv_to_base64(self.step_8_features),     # 7 (Feature Vis)
            self.cv_to_base64(self.final_result_img)     # 8
        ]
        
        r = self.analysis_report
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Laporan Detil Klasifikasi Bumbu</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .summary-box {{ background-color: #e8f4fc; padding: 20px; border-radius: 8px; border-left: 5px solid #3498db; margin-bottom: 30px; }}
                .result-title {{ font-size: 32px; font-weight: bold; color: #e74c3c; margin: 0; }}
                .logic-text {{ font-size: 16px; font-style: italic; color: #555; margin-top: 5px; }}
                
                .step-card {{ margin-bottom: 40px; }}
                .step-header {{ background-color: #34495e; color: white; padding: 10px 15px; font-size: 18px; border-radius: 5px 5px 0 0; }}
                .step-content {{ border: 1px solid #ddd; border-top: none; padding: 20px; display: flex; gap: 20px; align-items: flex-start; }}
                
                table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                
                .img-container {{ flex: 1; text-align: center; }}
                .img-container img {{ max-width: 100%; max-height: 250px; border: 1px solid #999; box-shadow: 2px 2px 5px rgba(0,0,0,0.2); }}
                .text-container {{ flex: 1; }}
                .metric-value {{ font-weight: bold; color: #2980b9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div style="text-align: center;">
                    <h1>Laporan Analisis Citra Komputer</h1>
                    <p>Waktu Analisis: {datetime.datetime.now().strftime("%d %B %Y, %H:%M WIB")}</p>
                </div>

                <div class="summary-box">
                    <h3>KESIMPULAN AKHIR</h3>
                    <p class="result-title">{r['prediction']}</p>
                    <p class="logic-text">" {r['logic_path']} "</p>
                </div>

                <h2>Rincian Proses & Analisis</h2>

                <div class="step-card">
                    <div class="step-header">Tahap 1: Pre-processing (Noise Reduction)</div>
                    <div class="step-content">
                        <div class="img-container">
                            <img src="{imgs[0]}" alt="Asli"><br><small>Citra Asli</small>
                            <br><br>
                            <img src="{imgs[1]}" alt="Blur"><br><small>Median Blur</small>
                        </div>
                        <div class="text-container">
                            <h4>Analisis:</h4>
                            <p>Citra asli sering mengandung <i>noise</i> (bintik-bintik kecil) yang dapat mengganggu deteksi warna.</p>
                            <ul>
                                <li><b>Metode:</b> Median Blur (Kernel 5x5).</li>
                                <li><b>Tujuan:</b> Menghaluskan tekstur bumbu tanpa menghilangkan tepi (edge) objek terlalu banyak. Ini penting agar bintik putih pada cabai tidak dianggap sebagai bawang putih.</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="step-card">
                    <div class="step-header">Tahap 2: Segmentasi Warna (HSV)</div>
                    <div class="step-content">
                        <div class="img-container">
                            <img src="{imgs[2]}" alt="HSV V-Channel"><br><small>Channel V (Brightness)</small>
                        </div>
                        <div class="text-container">
                            <h4>Analisis Warna:</h4>
                            <p>Sistem mengonversi citra RGB ke HSV untuk memisahkan intensitas cahaya dari informasi warna.</p>
                            <table>
                                <tr><th>Target Warna</th><th>Deteksi Pixel</th><th>Persentase</th></tr>
                                <tr><td><b>Merah</b> (Cabai/Bawang Merah)</td><td>{r['pixels']['R']} px</td><td class="metric-value">{r['metrics']['pct_red']}%</td></tr>
                                <tr><td><b>Hijau</b> (Cabai Hijau)</td><td>{r['pixels']['G']} px</td><td class="metric-value">{r['metrics']['pct_green']}%</td></tr>
                                <tr><td><b>Putih</b> (Bawang Putih)</td><td>{r['pixels']['W']} px</td><td class="metric-value">{r['metrics']['pct_white']}%</td></tr>
                            </table>
                            <p><i>*Persentase dihitung relatif terhadap luas area objek yang terdeteksi.</i></p>
                        </div>
                    </div>
                </div>

                <div class="step-card">
                    <div class="step-header">Tahap 3: Analisis Geometri (Bentuk)</div>
                    <div class="step-content">
                        <div class="img-container">
                            <img src="{imgs[7]}" alt="Shape Analysis"><br><small>Bounding Box (Biru) & Convex Hull (Kuning)</small>
                        </div>
                        <div class="text-container">
                            <h4>Analisis Morfologi:</h4>
                            <p>Jika warna dominan adalah Merah, sistem membedakan Cabai dan Bawang Merah berdasarkan bentuk.</p>
                            <table>
                                <tr><th>Metrik</th><th>Nilai</th><th>Interpretasi</th></tr>
                                <tr>
                                    <td><b>Aspect Ratio</b><br>(Panjang / Lebar)</td>
                                    <td class="metric-value">{r['aspect_ratio']}</td>
                                    <td>
                                        > 1.5 : Cenderung Lonjong (Cabai)<br>
                                        ~ 1.0 : Cenderung Bulat (Bawang)
                                    </td>
                                </tr>
                                <tr>
                                    <td><b>Solidity</b><br>(Area / Hull)</td>
                                    <td class="metric-value">{r['solidity']}</td>
                                    <td>
                                        < 0.8 : Melengkung/Tidak Padat (Cabai)<br>
                                        > 0.9 : Padat/Cembung (Bawang)
                                    </td>
                                </tr>
                                <tr>
                                    <td><b>Circularity</b><br>(Kebulatan)</td>
                                    <td class="metric-value">{r['circularity']}</td>
                                    <td>
                                        Mendekati 1.0 = Lingkaran Sempurna.
                                    </td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>

                <div class="step-card">
                    <div class="step-header">Tahap 4: Hasil Visualisasi Akhir</div>
                    <div class="step-content">
                        <div class="img-container">
                            <img src="{imgs[8]}" style="max-height: 400px;" alt="Final Result">
                        </div>
                        <div class="text-container">
                            <h4>Keputusan Algoritma:</h4>
                            <ol>
                                <li>Cek Hijau: <b>{r['metrics']['pct_green']}%</b> (Threshold > 10%)</li>
                                <li>Cek Merah: <b>{r['metrics']['pct_red']}%</b> (Threshold > 40%)</li>
                                <li>Jika Merah Dominan, cek Rasio & Solidity.</li>
                                <li>Jika tidak ada warna kuat, diasumsikan Bawang Putih.</li>
                            </ol>
                            <p><b>Prediksi Final: {r['prediction']}</b></p>
                        </div>
                    </div>
                </div>

            </div>
        </body>
        </html>
        """
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Laporan", "Laporan_Analisis_Bumbu.html", "HTML Files (*.html)")
        if file_path:
            with open(file_path, 'w') as f:
                f.write(html)
            QMessageBox.information(self, "Sukses", f"Laporan lengkap disimpan di:\n{file_path}")

if __name__ == '__main__':
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    window = SpiceClassifierApp()
    window.show()
    sys.exit(app.exec_())