import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO
from collections import deque

def calculate_angle(keypoints, a, b, c):
    """Tính góc giữa ba điểm chính"""
    # Kiểm tra độ tin cậy của các điểm chính
    if all(keypoints[i, 2] > 0.5 for i in [a, b, c]):
        a_pos = keypoints[a, :2]
        b_pos = keypoints[b, :2]
        c_pos = keypoints[c, :2]
        
        # Tính vector
        vector_ba = a_pos - b_pos
        vector_bc = c_pos - b_pos
        
        # Tính độ dài vector và tránh chia cho 0
        norm_ba = np.linalg.norm(vector_ba)
        norm_bc = np.linalg.norm(vector_bc)
        if norm_ba == 0 or norm_bc == 0:
            return None
            
        # Tính góc sử dụng tích vô hướng
        cosine_angle = np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    return None

class PushUpCounter:
    def __init__(self, source=0, conf=0.5, output=None, no_gui=False):
        """Khởi tạo bộ đếm chống đẩy"""
        # Tải mô hình YOLOv11 pose estimation
        try:
            self.model = YOLO("yolo11n-pose.pt")
        except Exception as e:
            print(f"Lỗi khi tải mô hình YOLOv8: {e}")
            raise
        
        self.conf = conf
        self.source = source
        self.output_path = output
        self.no_gui = no_gui
        
        # Biến trạng thái chống đẩy
        self.push_up_count = 0
        self.position = "up"
        self.position_buffer = deque(maxlen=5)
        self.angle_history = deque(maxlen=10)
        self.consecutive_frames = {"up": 0, "down": 0}
        self.required_consecutive = 1  # Giảm để tăng độ nhạy
        
        # Ngưỡng góc được tinh chỉnh cho chống đẩy
        self.camera_angles = {
            "front": {"up": 140, "down": 110},  # Góc khuỷu tay: ~140° (up), ~110° (down)
            "side": {"up": 150, "down": 100},
            "diagonal": {"up": 145, "down": 105}
        }
        self.current_angle = "front"
        
    def detect_camera_angle(self, keypoints):
        """Xác định góc camera dựa trên khả năng nhìn thấy điểm chính"""
        # Sử dụng góc mặc định "front" để đơn giản
        return "front"
    
    def get_position(self, keypoints):
        """Xác định vị trí chống đẩy từ các điểm chính"""
        # Chỉ số điểm chính: 5: vai trái, 6: vai phải, 7: khuỷu tay trái, 
        # 8: khuỷu tay phải, 9: cổ tay trái, 10: cổ tay phải
        
        # Chỉ sử dụng góc khuỷu tay để tăng độ chính xác
        left_elbow_angle = calculate_angle(keypoints, 5, 7, 9)
        right_elbow_angle = calculate_angle(keypoints, 6, 8, 10)
        
        # Thu thập các góc hợp lệ
        angles = [angle for angle in [left_elbow_angle, right_elbow_angle] if angle is not None]
        
        if not angles:
            return None
            
        # Tính góc trung bình
        avg_angle = sum(angles) / len(angles)
        self.angle_history.append(avg_angle)
        
        # In log để theo dõi góc
        print(f"Góc khuỷu tay trung bình: {avg_angle:.2f}°")
        
        # Yêu cầu ít nhất 3 góc để tính trung bình động
        if len(self.angle_history) < 3:
            return None
            
        running_avg = sum(self.angle_history) / len(self.angle_history)
        
        # Lấy ngưỡng dựa trên góc camera
        thresholds = self.camera_angles[self.current_angle]
        
        # Xác định vị trí
        if running_avg < thresholds["down"]:
            print("Vị trí: down")
            return "down"
        elif running_avg > thresholds["up"]:
            print("Vị trí: up")
            return "up"
            
        return None
    
    def count_push_ups(self, frame):
        """Xử lý một khung hình và đếm chống đẩy"""
        try:
            # Giảm kích thước khung hình để tối ưu
            frame = cv2.resize(frame, (640, 480))
            
            # Chạy dự đoán YOLOv8
            results = self.model.predict(frame, conf=self.conf, verbose=False)
            
            # Tạo bản sao của khung hình
            annotated_frame = frame.copy()
            
            if results and len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                result = results[0]
                
                # Trích xuất các điểm chính nếu có
                if len(result.keypoints.data) > 0:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    
                    # Xác định góc camera
                    self.current_angle = self.detect_camera_angle(keypoints)
                    
                    # Xác định vị trí chống đẩy
                    current_position = self.get_position(keypoints)
                    
                    # Nếu phát hiện vị trí, thêm vào bộ đệm
                    if current_position:
                        self.position_buffer.append(current_position)
                        
                        # Xử lý nếu bộ đệm có đủ dữ liệu
                        if len(self.position_buffer) >= 3:
                            # Đếm số vị trí trong bộ đệm
                            pos_count = {"up": 0, "down": 0}
                            for pos in self.position_buffer:
                                pos_count[pos] += 1
                            
                            # Lấy vị trí chiếm ưu thế
                            buffer_position = max(pos_count, key=pos_count.get)
                            
                            # Cập nhật số khung liên tiếp
                            if buffer_position == self.position:
                                self.consecutive_frames[buffer_position] += 1
                            else:
                                self.consecutive_frames[buffer_position] = 1
                                self.consecutive_frames[self.position] = 0
                            
                            # Kiểm tra thay đổi vị trí
                            if (buffer_position != self.position and 
                                self.consecutive_frames[buffer_position] >= self.required_consecutive):
                                # Đếm chống đẩy khi chuyển từ "down" sang "up"
                                if buffer_position == "up" and self.position == "down":
                                    self.push_up_count += 1
                                    print(f"Đã đếm chống đẩy! Tổng: {self.push_up_count}")
                                    
                                # Cập nhật vị trí
                                self.position = buffer_position
                    
                    # Vẽ khung xương lên khung hình
                    annotated_frame = result.plot()
            
            # Thêm thông tin chống đẩy
            cv2.putText(annotated_frame, f"Location: {self.position}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Push up counts: {self.push_up_count}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Camera Angle: {self.current_angle}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return annotated_frame
        
        except Exception as e:
            print(f"Lỗi khi xử lý khung hình: {e}")
            return frame
    
    def run(self):
        """Chạy bộ đếm chống đẩy"""
        # Thử mở nguồn video
        cap = None
        for attempt in range(3):
            try:
                if isinstance(self.source, str) and self.source.isdigit():
                    cap = cv2.VideoCapture(int(self.source))
                else:
                    cap = cv2.VideoCapture(self.source)
                if cap.isOpened():
                    break
                print(f"Thử lại lần {attempt + 1}/3 để mở nguồn video...")
                time.sleep(1)
            except Exception as e:
                print(f"Lỗi khi mở nguồn video: {e}")
        
        if not cap or not cap.isOpened():
            print(f"Lỗi: Không thể mở nguồn video {self.source}")
            return
        
        # Lấy thuộc tính video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        
        # Tạo bộ ghi video
        out = None
        if self.output_path:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (640, 480))
            except Exception as e:
                print(f"Lỗi khi tạo file đầu ra {self.output_path}: {e}")
        
        print("Bộ đếm chống đẩy đã khởi động. Nhấn 'q' để thoát nếu sử dụng GUI.")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Không thể đọc khung hình. Kết thúc video hoặc lỗi nguồn.")
                    break
                
                # Xử lý khung hình
                processed_frame = self.count_push_ups(frame)
                
                # Ghi vào đầu ra
                if out:
                    out.write(processed_frame)
                
                # Hiển thị khung hình nếu không tắt GUI
                if not self.no_gui:
                    try:
                        cv2.imshow('Bộ đếm chống đẩy', processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Người dùng nhấn 'q' để thoát.")
                            break
                    except Exception as e:
                        print(f"Lỗi khi hiển thị khung hình: {e}. Tắt GUI và tiếp tục.")
                        self.no_gui = True
                
                # Kiểm soát tốc độ khung hình
                time.sleep(0.033)
        
        except KeyboardInterrupt:
            print("Người dùng dừng chương trình.")
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý: {e}")
        
        finally:
            # Dọn dẹp tài nguyên
            if cap:
                cap.release()
            if out:
                out.release()
            if not self.no_gui:
                cv2.destroyAllWindows()
            
            print(f"Tổng số lần chống đẩy được đếm: {self.push_up_count}")

def main():
    """Hàm chính để chạy chương trình"""
    parser = argparse.ArgumentParser(description='Bộ đếm chống đẩy sử dụng YOLOv8')
    parser.add_argument('--source', type=str, default='0', 
                        help='Đường dẫn đến file video hoặc chỉ số camera (mặc định: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Ngưỡng độ tin cậy (mặc định: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Đường dẫn file đầu ra (tùy chọn)')
    parser.add_argument('--no-gui', action='store_true',
                        help='Tắt hiển thị GUI')
    
    args = parser.parse_args()
    
    counter = PushUpCounter(
        source=args.source,
        conf=args.conf,
        output=args.output,
        no_gui=args.no_gui
    )
    
    counter.run()

if __name__ == "__main__":
    main()