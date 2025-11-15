from dataclasses import dataclass

@dataclass
class GuardConfig:
    """
    Cấu hình Guard: Chứa các tham số liên quan đến Guard và các yêu cầu của dataset.
    """
    enabled: bool = True  # Cho phép Guard hoạt động hay không
    mode: str = "soft"    # Chế độ Guard: "soft" hoặc "hard"
    use_activity_guard: bool = True  # Sử dụng Guard kiểm tra hoạt động hay không
    use_trace_guard: bool = False    # Sử dụng Guard kiểm tra trace hay không
    penalty_factor: float = 0.5  # Hệ số phạt cho các ứng viên không hợp lệ
    min_prefix_len: int = 1  # Độ dài tối thiểu của prefix
    activity_dataset: str = "datasets/A-SAD_instructions.csv"  # Dữ liệu về hoạt động để kiểm tra Guard
    trace_dataset: str = "datasets/T-SAD_instructions.csv"  # Dữ liệu về trace để kiểm tra Guard
