import random
from typing import List, Dict, Any
from datasets import Dataset

# ----------------------------- Data Augmentation Functions ----------------------------- #

def augment_negative_samples(dataset: Dataset, label_column: str = "ds_labels", target_label: str = "False", augmentation_factor: int = 2) -> Dataset:
    """
    Tăng cường dữ liệu cho các mẫu có nhãn tiêu cực (anomalous).
    Tạo thêm các mẫu cho các mẫu có nhãn "False" (anomalous).

    Args:
        dataset: Dataset gốc.
        label_column: Tên cột chứa nhãn (ví dụ: "ds_labels").
        target_label: Nhãn mục tiêu cần tăng cường (mặc định là "False").
        augmentation_factor: Số lượng tăng cường cho mỗi mẫu tiêu cực.

    Returns:
        Dataset đã được tăng cường.
    """
    negative_samples = dataset.filter(lambda x: x[label_column] == target_label)
    augmented_samples = []

    for sample in negative_samples:
        for _ in range(augmentation_factor):
            augmented_samples.append(sample)

    # Thêm dữ liệu đã tăng cường vào dataset gốc
    augmented_dataset = dataset.concatenate(Dataset.from_dict(augmented_samples))
    return augmented_dataset


def random_sampling(dataset: Dataset, label_column: str = "ds_labels", target_label: str = "True", sample_size: int = 1000) -> Dataset:
    """
    Lấy mẫu ngẫu nhiên từ dataset theo nhãn mục tiêu (ví dụ: "True" hoặc "False").
    
    Args:
        dataset: Dataset gốc.
        label_column: Tên cột chứa nhãn (ví dụ: "ds_labels").
        target_label: Nhãn mục tiêu cần lấy mẫu (mặc định là "True").
        sample_size: Số lượng mẫu cần lấy.

    Returns:
        Dataset chứa các mẫu ngẫu nhiên.
    """
    target_samples = dataset.filter(lambda x: x[label_column] == target_label)
    sampled_data = target_samples.shuffle(seed=42).select(range(sample_size))
    return sampled_data


def generate_synthetic_data(dataset: Dataset, transformation_func: Any) -> Dataset:
    """
    Sinh ra dữ liệu tổng hợp (synthetic data) bằng cách áp dụng một hàm biến đổi.
    
    Args:
        dataset: Dataset gốc.
        transformation_func: Hàm dùng để tạo dữ liệu tổng hợp từ mẫu gốc.

    Returns:
        Dataset mới chứa dữ liệu tổng hợp.
    """
    augmented_samples = []

    for sample in dataset:
        new_sample = transformation_func(sample)  # Áp dụng hàm biến đổi
        augmented_samples.append(new_sample)

    augmented_dataset = Dataset.from_dict(augmented_samples)
    return augmented_dataset


def transform_sample_to_synthetic(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Một ví dụ về hàm biến đổi để tạo dữ liệu tổng hợp từ mẫu gốc.
    Có thể áp dụng các biến đổi như đảo ngược, thay đổi cấu trúc hoạt động, v.v.

    Args:
        sample: Mẫu gốc.

    Returns:
        Mẫu tổng hợp mới.
    """
    new_sample = sample.copy()
    # Ví dụ: Đảo ngược các hoạt động trong trace hoặc prefix
    if "prefix" in sample:
        new_sample["prefix"] = sample["prefix"][::-1]  # Đảo ngược trace hoặc prefix
    return new_sample
