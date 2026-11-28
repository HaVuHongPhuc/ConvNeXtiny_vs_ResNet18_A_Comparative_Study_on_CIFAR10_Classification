"""Giao diện Streamlit để so sánh dự đoán CIFAR-10 trên cùng một ảnh."""

import csv
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.datasets import CIFAR10

from src.cifar_compare.inference import (
    CLASS_NAMES,
    MODEL_NAMES,
    load_demo_model,
    predict_logits,
    prepare_demo_image,
    read_demo_runs,
)
from src.cifar_compare.utils import load_config, resolve_device


ROOT = Path(__file__).resolve().parent
MAX_IMAGE_MB = 10
MAX_IMAGE_PIXELS = 16_000_000


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_model(run, device, file_signature):
    """Dùng lại model trong cùng tiến trình, với phiên bản file là một phần khóa cache."""
    return load_demo_model(run, device)


@st.cache_resource(show_spinner=False)
def load_test_set(data_root):
    dataset = CIFAR10(data_root, train=False, download=False)
    if tuple(dataset.classes) != CLASS_NAMES:
        raise ValueError("Thứ tự lớp của dữ liệu mẫu không khớp CIFAR-10.")
    return dataset


def read_uploaded_image(upload):
    try:
        with Image.open(BytesIO(upload.getvalue()), formats=["JPEG", "PNG", "WEBP"]) as decoded:
            if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
                raise ValueError("Ảnh vượt quá 16 triệu điểm ảnh.")
            if getattr(decoded, "is_animated", False):
                raise ValueError("Hãy chọn một ảnh tĩnh.")
            return ImageOps.exif_transpose(decoded).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise ValueError("Không đọc được ảnh. Hãy dùng một ảnh JPG, PNG hoặc WebP hợp lệ.") from error


def ranked_classes(logits):
    order = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
    return [
        {"Hạng": rank, "Lớp": CLASS_NAMES[index], "Logit": str(logits[index])}
        for rank, index in enumerate(order, start=1)
    ]


def display_prediction(run, ranking, true_index):
    best = ranking[0]["Lớp"]
    with st.container(border=True):
        st.subheader(MODEL_NAMES[run.model_name])
        st.caption(f"Checkpoint epoch {run.epoch} · Seed {run.seed}")
        st.metric("Nhãn dự đoán", best)
        if true_index is not None:
            if best == CLASS_NAMES[true_index]:
                st.success("Khớp nhãn thật")
            else:
                st.warning(f"Nhãn thật: {CLASS_NAMES[true_index]}")
        st.markdown("**Ba lớp đứng đầu**")
        st.markdown("\n".join(f"{row['Hạng']}. {row['Lớp']}" for row in ranking[:3]))
    return best


def display_test_reference(runs):
    result_path = ROOT / "artifacts" / "comparison_table.csv"
    if not result_path.is_file():
        return
    try:
        with result_path.open(encoding="utf-8-sig", newline="") as file:
            results = {row["Model"]: row for row in csv.DictReader(file)}
        table = []
        for name, label in MODEL_NAMES.items():
            run, row = runs[name], results[name]
            if (
                row["Run"] != run.run_name
                or int(row["Best epoch"]) != run.epoch
                or row["best.pt SHA256 (12 ký tự đầu)"].lower() != run.sha256[:12]
            ):
                return
            table.append({
                "Mô hình": label,
                "Accuracy test": f'{float(row["Test Accuracy"]):.2%}',
                "Macro-F1": f'{float(row["Macro-F1"]):.2%}',
                "Latency trung bình": f'{float(row["Latency/batch 32 mean (ms)"]):.2f} ms/batch',
            })
    except (OSError, KeyError, ValueError):
        return
    with st.expander("Kết quả thực nghiệm trên 10.000 ảnh test"):
        st.table(table)
        st.caption("Kết quả đã lưu từ notebook 06. Latency là forward trên RTX 5070 Ti, batch 32.")


def main():
    st.set_page_config(page_title="So sánh mô hình CIFAR-10", page_icon=":material/compare:", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { max-width: 76rem; padding-top: 2.8rem; }
        h1 { letter-spacing: -0.035em; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("So sánh dự đoán CIFAR-10")
    st.write("Đưa cùng một ảnh qua ResNet18 và ConvNeXt-Tiny để xem hai mô hình dự đoán lớp nào.")

    try:
        runs = read_demo_runs(ROOT)
        config = load_config(ROOT / "configs" / "default.json", {})
    except (OSError, KeyError, ValueError) as error:
        st.error(f"Không đọc được cấu hình demo. {error}")
        st.stop()

    with st.sidebar:
        st.header("Thiết bị và checkpoint")
        device_choice = st.radio("Thiết bị suy luận", ["auto", "cpu"], format_func={"auto": "Tự động", "cpu": "CPU"}.get)
        device = resolve_device(device_choice)
        st.caption(f"Thiết bị đang chọn: {str(device).upper()}")
        for name, label in MODEL_NAMES.items():
            run = runs[name]
            st.markdown(f"**{label}**")
            st.caption(f"Epoch {run.epoch} · Seed {run.seed} · SHA-256 {run.sha256[:12]}")

    input_size = runs["resnet18"].input_size
    st.caption(f"RGB → Resize {input_size} × {input_size} → Chuẩn hóa ImageNet → Hai mô hình → 10 lớp")
    left, right = st.columns([1, 2.05], gap="large")
    image = None
    true_index = None
    with left:
        st.header("Ảnh đầu vào")
        source = st.radio("Nguồn ảnh", ["Ảnh mẫu CIFAR-10", "Tải ảnh lên"])
        if source == "Ảnh mẫu CIFAR-10":
            data_root = ROOT / config.get("data_root", "data/raw")
            try:
                test_set = load_test_set(str(data_root))
                index = st.number_input(
                    "Chỉ số ảnh test", min_value=0, max_value=len(test_set) - 1,
                    value=456, step=1, help="Thử ảnh 456, 3977 và 9227 để đối chiếu Phụ lục D.",
                )
                image, true_index = test_set[int(index)]
            except (RuntimeError, ValueError, OSError):
                st.warning("Chưa đọc được tập test CIFAR-10 tại data/raw. Bạn có thể tải một ảnh lên để thử.")
        else:
            upload = st.file_uploader(
                "Ảnh tĩnh JPG, PNG hoặc WebP", type=["jpg", "jpeg", "png", "webp"],
                max_upload_size=MAX_IMAGE_MB,
            )
            if upload is not None:
                try:
                    image = read_uploaded_image(upload)
                except ValueError as error:
                    st.error(str(error))
        if image is not None:
            if max(image.size) <= 32:
                preview = ImageOps.contain(image, (256, 256), Image.Resampling.NEAREST)
            else:
                preview = ImageOps.contain(image, (480, 350), Image.Resampling.BILINEAR)
            st.image(preview, width="stretch")
            st.caption(f"Ảnh gốc {image.width} × {image.height} px · RGB")
            if true_index is not None:
                st.markdown(f"**Nhãn thật** · {CLASS_NAMES[true_index]}")
        submit = st.button("Chạy dự đoán", type="primary", width="stretch", disabled=image is None)

    with right:
        st.header("Kết quả dự đoán")
        if submit and image is not None:
            try:
                with st.spinner("Đang nạp checkpoint và dự đoán…"):
                    batch = prepare_demo_image(image, input_size, device)
                    rankings = {}
                    for name, run in runs.items():
                        stat = run.checkpoint.stat()
                        model = cached_model(run, str(device), (stat.st_size, stat.st_mtime_ns))
                        rankings[name] = ranked_classes(predict_logits(model, batch))
            except FileNotFoundError:
                st.error("Thiếu best.pt. Hãy đặt cả hai checkpoint ở đúng đường dẫn trong reports/run_manifest.csv.")
            except (OSError, KeyError, ValueError, RuntimeError) as error:
                st.error(f"Suy luận chưa hoàn tất. {error}")
            else:
                panels = st.columns(2)
                predicted = []
                for panel, name in zip(panels, MODEL_NAMES):
                    with panel:
                        predicted.append(display_prediction(runs[name], rankings[name], true_index))
                if predicted[0] == predicted[1]:
                    st.info(f"Hai mô hình cùng dự đoán {predicted[0]}.")
                else:
                    st.info("Hai mô hình dự đoán hai lớp khác nhau.")
                with st.expander("Thứ hạng và logit của 10 lớp"):
                    st.caption("Logit là điểm đầu ra thô. Đọc thứ hạng lớp trong từng mô hình.")
                    columns = st.columns(2)
                    for column, name in zip(columns, MODEL_NAMES):
                        with column:
                            st.markdown(f"**{MODEL_NAMES[name]}**")
                            st.table(rankings[name])
        else:
            with st.container(border=True):
                st.write("Chọn ảnh rồi bấm **Chạy dự đoán** để xem hai kết quả cạnh nhau.")
        st.caption("Mỗi lượt là dự đoán cho một ảnh. Accuracy được tổng hợp trên 10.000 ảnh test.")

    st.divider()
    display_test_reference(runs)


if __name__ == "__main__":
    main()

