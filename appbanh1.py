import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import torch
import tensorflow as tf
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class BakeryCheckoutApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Nhận Diện Bánh & Thanh Toán Tiện Lợi")
        self.root.geometry("1300x850")
        self.root.resizable(False, False)
        self.root.configure(bg="#F0F2F5")

        # ---- Biến hệ thống ----
        self.cap = None
        self.current_image = None
        self.is_camera_on = False
        self.after_id = None
        self.last_thumbnails = []
        self.thumbnail_imgs = []  # giữ reference ảnh để tránh bị garbage collect

        # ---- Cấu hình AI ----
        self.CONFIDENCE_THRESHOLD = 0.55  # << ngưỡng tự tin tối thiểu, chỉnh theo thực tế
        self.DEBUG = True  # << bật/tắt log debug ra console

        # STANDARD_CAKE_AREA được hiệu chỉnh ở độ phân giải tham chiếu dưới đây.
        # !!! QUAN TRỌNG: đổi 2 số này thành đúng độ phân giải ảnh bạn dùng lúc đo ra
        # con số 22000 ban đầu (vd: ảnh webcam test lúc đó là 640x480, 1280x720, ...)
        self.REFERENCE_WIDTH = 1920
        self.REFERENCE_HEIGHT = 1080
        self.STANDARD_CAKE_AREA = 22000

        # << Kích thước input của model CNN - phải khớp với InputLayer trong config.json
        # của file .keras (model mới của bạn là 224, model cũ trước đó là 280)
        self.CNN_INPUT_SIZE = 224

        # ---- Dữ liệu bảng giá ----
        self.PRICE_DICT = {
            'banh-chuoi-nuong': 19000, 'banh-da-lon': 23000, 'banh-dua-luoi': 15000,
            'banh_mi_bo': 18000, 'cha-bong-cay': 27000, 'cookies_dua': 23000,
            'croissant': 30000, 'eggtart': 21000, 'muffin-viet-quat': 25000, 'patechaud': 30000
        }
        self.DISPLAY_NAME = {
            'banh-chuoi-nuong': 'Bánh Chuối Nướng', 'banh-da-lon': 'Bánh Da Lợn',
            'banh-dua-luoi': 'Bánh Dừa Lưới', 'banh_mi_bo': 'Bánh Mì Bò',
            'cha-bong-cay': 'Chà Bông Cay', 'cookies_dua': 'Cookies Dừa',
            'croissant': 'Croissant', 'eggtart': 'Egg Tart',
            'muffin-viet-quat': 'Muffin Việt Quất', 'patechaud': 'Pate Chaud'
        }

        self.setup_ui()
        self.root.update()

        # ---- LOAD MÔ HÌNH AI (Khởi tạo 1 lần duy nhất) ----
        self.status_label.config(text="● Đang tải mô hình AI...", fg="#F39C12")
        self.root.update()
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
            sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam, points_per_side=32, pred_iou_thresh=0.90,
                stability_score_thresh=0.94, crop_n_layers=0, min_mask_region_area=8000
            )

            self.cnn_model = self.load_cnn_model_safely("PhanLoaiBanhABC.h5")
            self.CLASS_NAMES = [
                'banh-chuoi-nuong', 'banh-da-lon', 'banh-dua-luoi', 'banh_mi_bo', 'cha-bong-cay',
                'cookies_dua', 'croissant', 'eggtart', 'muffin-viet-quat', 'patechaud'
            ]
            self.status_label.config(text="● Hệ thống Sẵn sàng", fg="#2ECC71")
        except Exception as e:
            messagebox.showerror(
                "Lỗi AI",
                "Không thể tải mô hình.\n\n"
                f"Lỗi: {e}\n\n"
                "Hãy xem console (terminal) để đọc traceback chi tiết của 3 cách thử load model.\n"
                "Thường cần chạy:\n"
                "  pip install --upgrade tensorflow keras\n"
                "rồi chạy lại chương trình."
            )
            self.status_label.config(text="● Lỗi Mô hình", fg="#E74C3C")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ======================================================
    #             LOAD MODEL .keras AN TOÀN (NHIỀU FALLBACK)
    # ======================================================
    def load_cnn_model_safely(self, model_path):
        """
        Thử nhiều cách load model .keras khác nhau, vì lỗi load model thường do:
        1. TF cũ không hỗ trợ định dạng Keras 3 (.keras) -> cần import keras chuẩn (keras>=3) thay vì tf.keras
        2. Model có custom layer/optimizer cần compile=False hoặc safe_mode=False
        3. Model lưu bằng phiên bản keras mới hơn bản đang cài
        In ra traceback đầy đủ của TỪNG lần thử để biết chính xác lỗi gốc là gì.
        """
        import traceback
        last_error = None

        # --- Cách 1: tf.keras.models.load_model (mặc định) ---
        try:
            print("[LOAD MODEL] Thử cách 1: tf.keras.models.load_model(...)")
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            last_error = e
            print("[LOAD MODEL] Cách 1 thất bại:")
            traceback.print_exc()

        # --- Cách 2: tf.keras.models.load_model với compile=False, safe_mode=False ---
        try:
            print("[LOAD MODEL] Thử cách 2: tf.keras.models.load_model(compile=False, safe_mode=False)")
            return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as e:
            last_error = e
            print("[LOAD MODEL] Cách 2 thất bại:")
            traceback.print_exc()

        # --- Cách 3: dùng package `keras` độc lập (keras 3) thay vì tf.keras ---
        try:
            print("[LOAD MODEL] Thử cách 3: import keras (gói keras độc lập, không qua tensorflow)")
            import keras
            print(f"[LOAD MODEL] Phiên bản keras độc lập: {keras.__version__}")
            return keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as e:
            last_error = e
            print("[LOAD MODEL] Cách 3 thất bại:")
            traceback.print_exc()

        # Nếu cả 3 cách đều fail -> raise lỗi cuối cùng để báo ra UI,
        # nhưng toàn bộ traceback chi tiết đã được in ra console ở trên để debug.
        raise last_error


    def setup_ui(self):
        left_frame = tk.Frame(self.root, bg="#F0F2F5")
        left_frame.place(x=20, y=20, width=800, height=800)

        self.screen_label = tk.Label(left_frame, bg="#1C1C1C", text="MÀN HÌNH CAMERA",
                                      fg="white", font=("Arial", 20))
        self.screen_label.place(x=0, y=0, width=800, height=550)

        self.status_label = tk.Label(left_frame, text="● Đang khởi động...", fg="#F39C12",
                                      bg="#1C1C1C", font=("Arial", 10, "bold"))
        self.status_label.place(x=10, y=8)

        btn_frame = tk.Frame(left_frame, bg="#F0F2F5")
        btn_frame.place(x=0, y=570, width=800, height=50)

        btn_style = {"font": ("Arial", 12, "bold"), "fg": "white", "cursor": "hand2", "bd": 0, "activebackground": "#444"}
        tk.Button(btn_frame, text="Mở Camera", bg="#0D6EFD", command=self.open_camera, **btn_style).place(x=0, y=0, width=180, height=45)
        tk.Button(btn_frame, text="Tắt Camera", bg="#6C757D", command=self.close_camera, **btn_style).place(x=200, y=0, width=180, height=45)
        tk.Button(btn_frame, text="Tải Ảnh Lên", bg="#198754", command=self.upload_image, **btn_style).place(x=400, y=0, width=180, height=45)
        tk.Button(btn_frame, text="Xoay 90°", bg="#FD7E14", command=self.rotate_image, **btn_style).place(x=600, y=0, width=180, height=45)

        self.thumbnail_frame = tk.Frame(left_frame, bg="white", relief=tk.SUNKEN, bd=2)
        self.thumbnail_frame.place(x=0, y=640, width=800, height=150)
        self.thumbnail_placeholder = tk.Label(self.thumbnail_frame, text="Khu vực hiển thị từng loại bánh sau khi cắt...", bg="white", fg="gray")
        self.thumbnail_placeholder.pack(pady=60)

        right_frame = tk.Frame(self.root, bg="white", relief=tk.RAISED, bd=2)
        right_frame.place(x=840, y=20, width=430, height=800)

        self.scan_btn = tk.Button(right_frame, text="QUÉT KHAY THỨC ĂN", bg="#E74C3D", fg="white",
                                   font=("Arial", 16, "bold"), cursor="hand2", bd=0, activebackground="#c0392b", command=self.scan_tray)
        self.scan_btn.place(x=20, y=20, width=390, height=60)

        tk.Label(right_frame, text="HÓA ĐƠN TẠM TÍNH", font=("Arial", 14, "bold"), bg="white", fg="#2C3E50").place(x=20, y=100)

        self.bill_text = tk.Text(right_frame, font=("Courier New", 11), bg="#F8F9FA", bd=0, state="disabled")
        self.bill_text.place(x=20, y=140, width=390, height=350)
        self.clear_bill()

        # Khung dọc dành cho ảnh QR tĩnh
        self.qr_label = tk.Label(right_frame, bg="white", text="(Chưa có hoá đơn)", fg="gray", font=("Arial", 9))
        self.qr_label.place(x=115, y=500, width=200, height=280)

    # ======================================================
    #                 XỬ LÝ CAMERA / ẢNH
    # ======================================================
    def open_camera(self):
        if self.is_camera_on: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không tìm thấy camera!")
            self.cap = None
            return
        self.is_camera_on = True
        self.status_label.config(text="● Camera đang chạy", fg="#2ECC71")
        self.update_frame()

    def update_frame(self):
        if not self.is_camera_on or self.cap is None or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if ret:
            self.current_image = frame
            self.display_image(frame)
        self.after_id = self.root.after(15, self.update_frame)

    def close_camera(self):
        self.is_camera_on = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.screen_label.configure(image='', text="MÀN HÌNH CAMERA")
        self.screen_label.image = None
        self.status_label.config(text="● Sẵn sàng", fg="#2ECC71")

    def upload_image(self):
        self.close_camera()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path: return
        frame = cv2.imread(file_path)
        if frame is not None:
            self.current_image = frame
            self.status_label.config(text="● Đã tải ảnh", fg="#3498DB")
            self.display_image(frame)

    def rotate_image(self):
        if self.current_image is not None and not self.is_camera_on:
            self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_CLOCKWISE)
            self.display_image(self.current_image)

    def display_image(self, frame):
        cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)
        target_w, target_h = 800, 550
        img_w, img_h = pil_img.size
        scale = min(target_w / img_w, target_h / img_h)
        new_size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (target_w, target_h), (28, 28, 28))
        offset = ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2)
        canvas.paste(pil_img, offset)
        imgtk = ImageTk.PhotoImage(image=canvas)
        self.screen_label.imgtk = imgtk
        self.screen_label.configure(image=imgtk, text="")

    # ======================================================
    #                 XỬ LÝ HOÁ ĐƠN / AI CORE
    # ======================================================
    def _set_bill_text(self, text: str):
        self.bill_text.config(state="normal")
        self.bill_text.delete(1.0, tk.END)
        self.bill_text.insert(tk.END, text)
        self.bill_text.config(state="disabled")

    def clear_bill(self):
        header = f"{'TÊN MÓN'.ljust(24)}| {'SL'.ljust(3)}| {'GIÁ'}\n" + "-" * 42 + "\n"
        self._set_bill_text(header)

    def generate_qr(self):
        try:
            img_qr = Image.open("qr_thanh_toan.jpg").convert("RGB")
            img_qr = img_qr.resize((200, 280), Image.Resampling.LANCZOS)
            imgtk_qr = ImageTk.PhotoImage(image=img_qr)
            self.qr_label.imgtk = imgtk_qr
            self.qr_label.configure(image=imgtk_qr, text="")
        except Exception as e:
            self.qr_label.configure(text="Không tìm thấy\nảnh qr_thanh_toan.jpg", image="", fg="red")

    # ---- HIỂN THỊ ẢNH ĐÃ CẮT (THUMBNAIL) ----
    def render_thumbnails(self):
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        self.thumbnail_imgs = []

        if not self.last_thumbnails:
            ph = tk.Label(self.thumbnail_frame, text="Không phát hiện vùng bánh nào...", bg="white", fg="gray")
            ph.pack(pady=60)
            return

        # Frame cuộn ngang đơn giản (nếu nhiều ảnh sẽ tự co lại)
        container = tk.Frame(self.thumbnail_frame, bg="white")
        container.pack(fill="both", expand=True)

        for item in self.last_thumbnails:
            cell = tk.Frame(container, bg="white", relief=tk.GROOVE, bd=1)
            cell.pack(side="left", padx=4, pady=8)

            pil_img = Image.fromarray(item["image"]).resize((90, 90), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil_img)
            self.thumbnail_imgs.append(imgtk)  # giữ reference

            tk.Label(cell, image=imgtk, bg="white").pack(padx=4, pady=(4, 0))

            color = "#2ECC71" if item["confidence"] >= self.CONFIDENCE_THRESHOLD else "#E74C3C"
            tk.Label(cell, text=f'{item["label"]}', bg="white", font=("Arial", 8, "bold")).pack()
            tk.Label(cell, text=f'{item["confidence"]*100:.0f}%', bg="white",
                     fg=color, font=("Arial", 8)).pack(pady=(0, 4))

    def scan_tray(self):
        if self.current_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng mở Camera hoặc tải ảnh lên trước!")
            return

        if not hasattr(self, 'mask_generator') or not hasattr(self, 'cnn_model'):
            messagebox.showerror("Lỗi AI", "Mô hình AI chưa được tải thành công. Vui lòng kiểm tra lại các file .pth và .keras!")
            return

        self.scan_btn.config(state="disabled", text="AI ĐANG NHẬN DIỆN...")
        self.root.update_idletasks()

        try:
            detected_items = self.run_ai_detection(self.current_image)
            self.render_thumbnails()  # << luôn hiển thị thumbnail dù có hay không có item hợp lệ
            if not detected_items:
                messagebox.showinfo("Kết quả", "Không nhận diện được món nào trong khay (hoặc độ tự tin quá thấp)!")
                self.clear_bill()
            else:
                self.render_bill(detected_items)
        except Exception as e:
            messagebox.showerror("Lỗi nhận diện", f"Lỗi hệ thống AI: {e}")

        self.scan_btn.config(state="normal", text="QUÉT KHAY THỨC ĂN")

    def run_ai_detection(self, image):
        # 1. Chạy SAM tách khay
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_masks = self.mask_generator.generate(image_rgb)
        raw_masks = sorted(raw_masks, key=lambda x: x['area'], reverse=True)

        clean_masks = []
        img_height, img_width, _ = image.shape
        total_image_area = img_height * img_width
        max_area_allow = total_image_area * 0.15

        # << CHUẨN HÓA STANDARD_CAKE_AREA theo tỉ lệ % diện tích ảnh,
        #    để không bị lệch khi ảnh đầu vào có độ phân giải khác lúc calibrate
        reference_area = self.REFERENCE_WIDTH * self.REFERENCE_HEIGHT
        standard_cake_ratio = self.STANDARD_CAKE_AREA / reference_area
        standard_cake_area_scaled = standard_cake_ratio * total_image_area

        if self.DEBUG:
            print(f"[DEBUG] Ảnh vào: {img_width}x{img_height}, "
                  f"standard_cake_area_scaled={standard_cake_area_scaled:.0f}, "
                  f"so_mask_tho={len(raw_masks)}")

        # 2. Lọc vùng hợp lệ
        for mask_data in raw_masks:
            area = mask_data['area']
            x, y, w, h = [int(v) for v in mask_data['bbox']]
            aspect_ratio = w / float(h)

            if aspect_ratio > 2.2 or aspect_ratio < 0.45: continue
            if area < 15000 or area > max_area_allow: continue

            is_duplicate = False
            current_mask = mask_data['segmentation']
            for saved_mask in clean_masks:
                intersection = np.logical_and(current_mask, saved_mask['segmentation'])
                if np.sum(intersection) / np.sum(current_mask) > 0.4:
                    is_duplicate = True
                    break
            if not is_duplicate:
                clean_masks.append(mask_data)

        clean_masks = sorted(clean_masks, key=lambda x: (x['bbox'][1], x['bbox'][0]))

        if self.DEBUG:
            print(f"[DEBUG] Sau khi lọc còn {len(clean_masks)} vùng hợp lệ")

        # 3. Chuẩn bị Batch đưa vào CNN
        cnn_input_batch = []
        mask_areas = []
        crop_thumbnails = []

        for mask_data in clean_masks:
            x, y, w, h = [int(v) for v in mask_data['bbox']]
            area = mask_data['area']

            pad_w, pad_h = int(w * 0.05), int(h * 0.05)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(img_width, x + w + pad_w), min(img_height, y + h + pad_h)

            cropped_natural = image_rgb[y1:y2, x1:x2]

            # << Model mới nhận input 224x224 và TỰ rescaling (1/255) bên trong
            # (xem layer Rescaling trong config.json) -> KHÔNG chia /255 thủ công nữa,
            # nếu không sẽ bị chia 255 hai lần làm ảnh gần như toàn số 0.
            img_for_cnn = cv2.resize(cropped_natural, (self.CNN_INPUT_SIZE, self.CNN_INPUT_SIZE))
            img_array = tf.keras.preprocessing.image.img_to_array(img_for_cnn)

            cnn_input_batch.append(img_array)
            mask_areas.append(area)
            crop_thumbnails.append(cropped_natural)

        detected_items = {}
        self.last_thumbnails = []

        # 4. Dự đoán và gom nhóm kết quả
        if len(cnn_input_batch) > 0:
            cnn_input_batch = np.array(cnn_input_batch)
            all_predictions = self.cnn_model.predict(cnn_input_batch, batch_size=32, verbose=0)

            for idx in range(len(cnn_input_batch)):
                score = all_predictions[idx]
                class_idx = np.argmax(score)
                confidence = float(score[class_idx])
                predicted_label = self.CLASS_NAMES[class_idx]
                actual_area = mask_areas[idx]

                # Lưu lại để hiển thị thumbnail (kể cả khi confidence thấp, để bạn debug bằng mắt)
                self.last_thumbnails.append({
                    "image": crop_thumbnails[idx],
                    "label": self.DISPLAY_NAME.get(predicted_label, predicted_label),
                    "confidence": confidence
                })

                if self.DEBUG:
                    top3_idx = np.argsort(score)[-3:][::-1]
                    top3_str = [(self.CLASS_NAMES[i], f"{score[i]:.2f}") for i in top3_idx]
                    print(f"[DEBUG] vùng#{idx}: area={actual_area}, pred={predicted_label} "
                          f"({confidence:.2f}), top3={top3_str}")

                # << BỎ QUA nếu model không đủ tự tin
                if confidence < self.CONFIDENCE_THRESHOLD:
                    continue

                # Logic đặc biệt cho Bánh Da Lợn (đã chuẩn hóa theo % diện tích ảnh)
                if predicted_label == 'banh-da-lon':
                    calculated_count = int(np.round(actual_area / standard_cake_area_scaled))
                    cake_count = max(1, min(calculated_count, 6))  # << chặn trần tránh số ảo
                else:
                    cake_count = 1

                if predicted_label in detected_items:
                    detected_items[predicted_label] += cake_count
                else:
                    detected_items[predicted_label] = cake_count

        return {k: v for k, v in detected_items.items() if k in self.PRICE_DICT}

    def render_bill(self, detected_items: dict):
        lines = [f"{'TÊN MÓN'.ljust(24)}| {'SL'.ljust(3)}| {'GIÁ'}", "-" * 42]
        total_money = 0
        for item, count in detected_items.items():
            price = self.PRICE_DICT[item]
            subtotal = price * count
            total_money += subtotal
            name = self.DISPLAY_NAME.get(item, item.upper())
            name_str = name[:23].ljust(24)
            count_str = str(count).ljust(3)
            price_str = f"{subtotal:,}".replace(",", ".")
            lines.append(f"{name_str}| {count_str}| {price_str}")
        lines.append("-" * 42)
        lines.append("TỔNG CỘNG:")
        lines.append(f"{total_money:,} VNĐ".replace(",", ".").rjust(42))
        self._set_bill_text("\n".join(lines) + "\n")
        self.generate_qr()

    def on_close(self):
        self.close_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = BakeryCheckoutApp(root)
    root.mainloop()