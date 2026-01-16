import pandas as pd
import logging
import os
import sys
from pathlib import Path

# 1. Cấu hình Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # In ra màn hình console
    ]
)
logger = logging.getLogger(__name__)

# Giả sử Schema mong đợi (Cấu hình trước)
EXPECTED_COLUMNS = ['id', 'name', 'amount', 'date']

def run_data_pipeline(input_path: str, output_path: str):
    logger.info("=== BẮT ĐẦU PIPELINE XỬ LÝ DỮ LIỆU ===")
    
    df = None

    try:
        # --- BƯỚC 1: LOAD CSV ---
        logger.info(f"[Step 1/4] Đang đọc file từ: {input_path}")
        
        # Kiểm tra file tồn tại thủ công (hoặc để pandas tự catch)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File không tồn tại: {input_path}")
            
        df = pd.read_csv(input_path)
        
        if df.empty:
            raise ValueError("File CSV rỗng, không có dữ liệu để xử lý.")
            
        logger.info(f"-> Đã load thành công {len(df)} dòng.")

        # --- BƯỚC 2: CLEAN DATA ---
        logger.info("[Step 2/4] Đang làm sạch dữ liệu...")
        
        # Ví dụ cleaning: Xóa khoảng trắng ở tên cột, xóa dòng duplicate
        df.columns = [c.strip() for c in df.columns] 
        initial_count = len(df)
        df = df.drop_duplicates()
        
        # Ví dụ cleaning: Điền giá trị 0 cho ô trống ở cột số (nếu có)
        if 'amount' in df.columns:
            df['amount'] = df['amount'].fillna(0)
            
        logger.info(f"-> Đã làm sạch. Loại bỏ {initial_count - len(df)} dòng trùng lặp.")

        # --- BƯỚC 3: VALIDATE SCHEMA ---
        logger.info("[Step 3/4] Đang kiểm tra cấu trúc (Validate Schema)...")
        
        # Kiểm tra xem các cột bắt buộc có tồn tại không
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        
        if missing_cols:
            # Đây là logic để raise lỗi Schema
            raise KeyError(f"Schema không hợp lệ. Thiếu các cột bắt buộc: {missing_cols}")
            
        # Kiểm tra kiểu dữ liệu (ví dụ amount phải là số)
        if not pd.api.types.is_numeric_dtype(df['amount']):
             raise TypeError("Cột 'amount' chứa dữ liệu không phải là số.")
             
        logger.info("-> Schema hợp lệ.")

        # --- BƯỚC 4: OUTPUT PARQUET ---
        logger.info(f"[Step 4/4] Đang lưu file Parquet tại: {output_path}")
        
        # Cần cài thư viện pyarrow hoặc fastparquet: pip install pyarrow
        df.to_parquet(output_path, index=False, engine='pyarrow')
        
        logger.info("-> Lưu file thành công.")
        logger.info("=== PIPELINE HOÀN THÀNH XUẤT SẮC ===")

    # --- XỬ LÝ 3 TRƯỜNG HỢP LỖI PHỔ BIẾN ---
    
    except FileNotFoundError as e:
        # Lỗi 1: Không tìm thấy file input
        logger.error(f"LỖI INPUT: {str(e)}")
        # Có thể thêm logic gửi email cảnh báo tại đây
        
    except (KeyError, TypeError, ValueError) as e:
        # Lỗi 2: Dữ liệu bẩn, sai schema, file rỗng
        logger.error(f"LỖI DỮ LIỆU/SCHEMA: {str(e)}")
        
    except Exception as e:
        # Lỗi 3: Các lỗi hệ thống khác (Permission denied, Out of memory...)
        logger.error(f"LỖI KHÔNG XÁC ĐỊNH: {str(e)}")
        
    finally:
        logger.info("Kết thúc phiên làm việc.\n")

# --- CHẠY THỬ ---
if __name__ == "__main__":
    # Tạo data giả để test (Bạn có thể bỏ qua phần này nếu đã có file thật)
    dummy_data = {
        'id': [1, 2, 3, 3], # Có duplicate
        'name': [' A', 'B ', 'C', 'C'],
        'amount': [100, None, 300, 300], # Có None
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03']
    }
    # Lưu file CSV tạm
    pd.DataFrame(dummy_data).to_csv(Path(__file__).parent / 'files' / 'data_input.csv', index=False)
    
    # Chạy pipeline
    run_data_pipeline(Path(__file__).parent / 'files' / 'data_input.csv', Path(__file__).parent / 'files' / 'data_output.parquet')
    
    # Test thử trường hợp lỗi (bỏ comment để chạy)
    # run_data_pipeline('file_khong_ton_tai.csv', 'out.parquet')