from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from core.export import export_field
from core.from_images import FromImagesConfig, invert_field_from_images
from core.generator import RandomFieldConfig, build_meta, generate_random_field
from core.io import get_output_root, write_npz
from core.visualization import (
    compute_slice_arrays_from_npz,
    render_slice_3d,
    render_field_and_mask_png,
    resolve_global_vmin_vmax,
)


APP_VERSION = "0.2.0"


def _resolve_output_dir() -> Path:
    output_dir = get_output_root("OrientationFieldGenerator")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


OUTPUT_DIR = _resolve_output_dir()
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_STYLE_CFG = {
    "dpi": 150,
    "figsize": (5.2, 4.2),
    "layout_main": [0.07, 0.07, 0.74, 0.86],
    "layout_side": [0.87, 0.1, 0.02, 0.8],
}
LOG_FILE = LOG_DIR / "gui.log"


TEXT = {
    "zh": {
        "app_title": "取向场生成系统",
        "welcome_title": "选择你的工作流",
        "welcome_subtitle": "模式选择",
        "welcome_random": "随机取向场",
        "welcome_random_desc": "使用随机晶粒分布创建新的取向场。",
        "welcome_images": "从图像反演",
        "welcome_images_desc": "从显微组织图像反演取向场并提取织构信息。",
        "btn_back": "上一步",
        "btn_next": "下一步",
        "btn_home": "返回首页",
        "btn_run": "生成/反演",
        "btn_export": "导出",
        "btn_browse": "选择",
        "btn_add": "添加",
        "btn_remove": "删除",
        "btn_normalize": "归一化权重",
        "btn_mid": "中间切片",
        "btn_start": "开始",
        "btn_stop": "停止",
        "group_geometry": "几何与网格",
        "group_topology": "晶粒拓扑",
        "group_absolute": "绝对取向（θ）",
        "group_gb": "晶界错配（Δθ）",
        "group_texture_advanced": "混合采样参数（可选）",
        "group_defect": "缺陷模型",
        "group_pbc": "周期边界",
        "group_output": "输出与复现",
        "group_inputs": "输入文件",
        "group_colorbar": "色条与映射",
        "group_mask": "缺陷与掩膜",
        "group_export": "导出",
        "label_dim": "维度",
        "label_shape_x": "网格点数 Nx",
        "label_shape_y": "网格点数 Ny",
        "label_shape_z": "网格点数 Nz",
        "label_grain_mean": "平均晶粒尺寸（像素/体素）",
        "label_grain_std": "晶粒尺寸标准差",
        "label_grain_dist": "晶粒尺寸分布",
        "label_theta_min": "取向角范围 θ_min/θ_max（度）",
        "label_theta_max": "取向角范围 θ_min/θ_max（度）",
        "label_theta_policy": "超界处理策略（正态采样）",
        "label_orientation_dist": "取向分布类型",
        "label_mean_theta": "正态均值 μ（度）",
        "label_orientation_std": "正态标准差 σ（度）",
        "label_gb_hard": "晶界错配角硬上限（度）",
        "label_gb_target": "晶界错配角目标（P95，可选）",
        "label_gb_sampler": "晶界错配角采样器",
        "label_gb_mean": "Δθ 均值（度）",
        "label_gb_std": "Δθ 标准差（度）",
        "label_gb_weight": "权重",
        "label_gb_name": "备注",
        "label_defect_enabled": "开启缺陷",
        "label_defect_fraction": "缺陷比例",
        "label_defect_shape": "缺陷形状",
        "label_defect_size_mean": "缺陷平均尺寸",
        "label_defect_size_std": "缺陷尺寸标准差",
        "label_defect_size_dist": "缺陷尺寸分布",
        "label_periodic": "周期边界",
        "label_seed": "随机种子 (可空)",
        "label_outdir": "输出目录",
        "label_prefix": "文件前缀",
        "label_field_image": "场图",
        "label_cbar_image": "色条",
        "label_cbar_min": "色条最小值（度）",
        "label_cbar_max": "色条最大值（度）",
        "label_cbar_direction": "色条方向",
        "label_defect_policy": "缺陷策略",
        "label_defect_threshold": "缺陷阈值",
        "label_color_tolerance": "颜色容差",
        "label_axis": "轴",
        "label_slice": "切片",
        "label_formats": "导出格式",
        "label_csv_axis": "CSV 轴",
        "label_csv_index": "CSV 切片",
        "label_status": "状态",
        "label_language": "语言",
        "label_language_zh": "中文",
        "label_language_en": "英文",
        "status_ready": "准备就绪",
        "status_running": "处理中…",
        "status_done": "完成",
        "status_error": "错误",
        "panel_logs": "日志",
        "dialog_render_title": "渲染中",
        "dialog_export_title": "导出中",
        "msg_validation_title": "校验",
        "msg_missing_title": "缺少",
        "msg_error_title": "错误",
        "msg_export_title": "导出",
        "msg_missing_config": "请先配置参数",
        "msg_missing_generate": "请先生成",
        "msg_missing_inputs": "请选择输入文件",
        "msg_missing_inputs_config": "请先配置输入",
        "msg_missing_invert": "请先反演",
        "msg_select_format": "至少选择一种格式",
        "msg_export_done": "导出完成",
        "msg_export_confirm": "将导出以下格式：{formats}\n导出目录：{outdir}\n输入文件：{input_npz}\n\n是否继续？",
        "msg_export_paths": "导出完成：\n{paths}",
        "msg_cbar_min_validation": "色条最小值必须小于最大值",
        "msg_rendering_preview": "正在生成与渲染预览…",
        "msg_rendering_slice": "正在渲染切片 {axis}={index}…",
        "msg_exporting": "正在导出…（可取消）",
        "msg_export_cancelled": "已取消导出",
        "msg_orientation_title": "取向",
        "msg_orientation_deprecated": "取向分布的混合模式在界面中已弃用，已回退到正态分布。",
        "msg_meta_random": "维度={dim} 形状={shape} 掩膜比例={mask_ratio:.3f} 种子={seed}",
        "msg_meta_invert": "维度={dim} 形状={shape} 掩膜比例={mask_ratio:.3f}",
        "value_mid": "中间",
        "value_optional": "（可选）",
        "default_prefix_random": "随机取向场",
        "default_prefix_invert": "图像反演",
        "table_gb_mean": "均值",
        "table_gb_std": "标准差",
        "table_gb_weight": "权重",
        "table_gb_note": "备注",
        "combo_orientation_random": "随机",
        "combo_orientation_normal": "正态",
        "combo_sampler_normal": "正态",
        "combo_sampler_mixture": "混合",
        "combo_theta_truncate": "截断",
        "combo_theta_clip": "裁剪",
        "combo_grain_uniform": "均匀",
        "combo_grain_normal": "正态",
        "combo_grain_lognormal": "对数正态",
        "combo_defect_uniform": "均匀",
        "combo_defect_normal": "正态",
        "combo_defect_lognormal": "对数正态",
        "combo_defect_sphere": "球体",
        "combo_defect_ellipsoid": "椭球",
        "combo_defect_blob": "不规则",
        "combo_defect_disk": "圆盘",
        "combo_defect_ellipse": "椭圆",
        "combo_defect_policy_nan": "空值",
        "combo_defect_policy_zero": "置零",
        "combo_defect_policy_fill": "填充",
        "combo_cbar_top_high": "顶部高值",
        "combo_cbar_top_low": "顶部低值",
        "error_theta_range": "取向角下限必须小于上限",
        "error_gb_hard": "晶界错配角硬上限必须大于 0",
        "error_defect_fraction": "缺陷比例必须在 [0,1]",
        "error_shape_positive": "网格尺寸必须为正",
        "error_shape_z_positive": "Z 方向网格尺寸必须为正",
        "error_grain_std": "晶粒尺寸标准差必须小于等于平均值",
        "error_grain_too_large": "晶粒过大，可能只剩极少晶粒",
        "error_mixture_format": "混合分布格式无效",
        "error_mixture_required": "混合分布输入不能为空",
        "tip_dim_2d": "二维取向场",
        "tip_dim_3d": "三维取向场",
        "tip_shape": "网格尺寸，需为正整数",
        "tip_grain_mean": "晶粒平均尺度（像素/体素单位），影响晶粒编号拓扑的平均晶粒大小",
        "tip_grain_std": "晶粒尺度离散程度（标准差）",
        "tip_grain_dist": "晶粒尺寸分布类型：均匀/正态/对数正态",
        "tip_shape_z": "仅 3D 使用，2D 时隐藏",
        "tip_orientation_dist": "绝对取向 θ 分布类型",
        "tip_mean_theta": "绝对取向 θ 均值（度）",
        "tip_orientation_std": "绝对取向 θ 标准差（度）",
        "tip_theta_range": "绝对取向 θ 范围（度）",
        "tip_theta_policy": "绝对取向 θ 超界策略",
        "tip_gb_hard": "晶界错配角 Δθ 硬上限",
        "tip_gb_target": "晶界错配角 Δθ 目标值 (P95)",
        "tip_gb_sampler": "晶界错配角 Δθ 采样器",
        "tip_texture_mixture": "该字符串定义晶界错配角 Δθ 的混合分布参数（均值/标准差/权重），不是绝对取向 θ",
        "tip_texture_table": "该表定义晶界错配角 Δθ 的混合分布参数（均值/标准差/权重），不是绝对取向 θ",
        "tip_defect_enabled": "是否开启缺陷/掩膜",
        "tip_defect_fraction": "缺陷比例 [0,1]",
        "tip_defect_size_mean": "缺陷尺寸均值",
        "tip_defect_size_std": "缺陷尺寸标准差",
        "tip_defect_size_dist": "缺陷尺寸分布",
        "tip_pbc_x": "启用 x 方向周期边界",
        "tip_pbc_y": "启用 y 方向周期边界",
        "tip_pbc_z": "启用 z 方向周期边界",
        "tip_outdir": "输出目录",
        "tip_prefix": "输出文件前缀",
        "tip_seed": "随机种子，留空自动生成",
        "tip_axis": "选择切片轴",
        "tip_slice_slider": "切片索引",
        "tip_slice_spin": "切片索引",
        "tip_mid": "跳转到中间切片",
        "tip_export_dir": "导出目录",
        "tip_export_npz": "导出 NPZ",
        "tip_export_json": "导出 JSON",
        "tip_export_csv": "导出 CSV",
        "tip_export_h5": "导出 HDF5",
        "tip_csv_axis": "3D CSV 切片轴",
        "tip_csv_index": "切片索引或中间",
        "tip_field_image": "输入场图像文件",
        "tip_cbar_image": "输入色条图像文件",
        "tip_cbar_min": "色条最小值（度）",
        "tip_cbar_max": "色条最大值（度）",
        "tip_cbar_direction": "色条方向（顶部高/顶部低）",
        "tip_defect_policy": "缺陷区域处理策略",
        "tip_color_tolerance": "颜色容差",
        "tip_defect_threshold": "缺陷阈值",
    },
    "en": {
        "app_title": "Orientation Field Generator GUI",
        "welcome_title": "Choose your workflow",
        "welcome_subtitle": "Mode Selection",
        "welcome_random": "Random Orientation Field",
        "welcome_random_desc": "Create a new orientation field with random grain distributions.",
        "welcome_images": "Inverse from Image",
        "welcome_images_desc": "Infer an orientation field from microstructure images and extract texture info.",
        "btn_back": "Back",
        "btn_next": "Next",
        "btn_home": "Home",
        "btn_run": "Run",
        "btn_export": "Export",
        "btn_browse": "Browse",
        "btn_add": "Add",
        "btn_remove": "Remove",
        "btn_normalize": "Normalize",
        "btn_mid": "Mid",
        "btn_start": "Start",
        "btn_stop": "Stop",
        "group_geometry": "Geometry/Grid",
        "group_topology": "Grain topology",
        "group_absolute": "Absolute orientation (θ)",
        "group_gb": "GB misorientation (Δθ)",
        "group_texture_advanced": "Mixture sampler (optional)",
        "group_defect": "Defect model",
        "group_pbc": "Periodic boundary",
        "group_output": "Output & reproducibility",
        "group_inputs": "Inputs",
        "group_colorbar": "Colorbar",
        "group_mask": "Defect & mask",
        "group_export": "Export",
        "label_dim": "Dimension",
        "label_shape_x": "Grid points Nx",
        "label_shape_y": "Grid points Ny",
        "label_shape_z": "Grid points Nz",
        "label_grain_mean": "Mean grain size (px/voxel)",
        "label_grain_std": "Grain size std",
        "label_grain_dist": "Grain size distribution",
        "label_theta_min": "θ range (deg)",
        "label_theta_max": "θ range (deg)",
        "label_theta_policy": "Out-of-range policy (normal sampler)",
        "label_orientation_dist": "Orientation distribution",
        "label_mean_theta": "Normal mean μ (deg)",
        "label_orientation_std": "Normal std σ (deg)",
        "label_gb_hard": "GB misorientation hard max (deg)",
        "label_gb_target": "GB target (P95, optional)",
        "label_gb_sampler": "GB misorientation sampler",
        "label_gb_mean": "Δθ mean (deg)",
        "label_gb_std": "Δθ std (deg)",
        "label_gb_weight": "Weight",
        "label_gb_name": "Note",
        "label_defect_enabled": "Enable defects",
        "label_defect_fraction": "Defect fraction",
        "label_defect_shape": "Defect shape",
        "label_defect_size_mean": "Defect size mean",
        "label_defect_size_std": "Defect size std",
        "label_defect_size_dist": "Defect size dist",
        "label_periodic": "Periodic boundary",
        "label_seed": "Seed (optional)",
        "label_outdir": "Output dir",
        "label_prefix": "Filename prefix",
        "label_field_image": "Field image",
        "label_cbar_image": "Colorbar image",
        "label_cbar_min": "Colorbar min (deg)",
        "label_cbar_max": "Colorbar max (deg)",
        "label_cbar_direction": "Colorbar direction",
        "label_defect_policy": "Defect policy",
        "label_defect_threshold": "Defect threshold",
        "label_color_tolerance": "Color tolerance",
        "label_axis": "Axis",
        "label_slice": "Slice",
        "label_formats": "Formats",
        "label_csv_axis": "CSV axis",
        "label_csv_index": "CSV index",
        "label_status": "Status",
        "label_language": "Language",
        "label_language_zh": "Chinese",
        "label_language_en": "English",
        "status_ready": "Ready",
        "status_running": "Processing…",
        "status_done": "Done",
        "status_error": "Error",
        "panel_logs": "Logs",
        "dialog_render_title": "Rendering",
        "dialog_export_title": "Exporting",
        "msg_validation_title": "Validation",
        "msg_missing_title": "Missing",
        "msg_error_title": "Error",
        "msg_export_title": "Export",
        "msg_missing_config": "Please configure parameters first",
        "msg_missing_generate": "Generate first",
        "msg_missing_inputs": "Select input files",
        "msg_missing_inputs_config": "Please configure inputs",
        "msg_missing_invert": "Invert first",
        "msg_select_format": "Select at least one format",
        "msg_export_done": "Export completed",
        "msg_export_confirm": "Formats: {formats}\nOutput directory: {outdir}\nInput file: {input_npz}\n\nContinue?",
        "msg_export_paths": "Export completed:\n{paths}",
        "msg_cbar_min_validation": "Colorbar min must be less than max",
        "msg_rendering_preview": "Generating and rendering preview…",
        "msg_rendering_slice": "Rendering slice {axis}={index}…",
        "msg_exporting": "Exporting… (cancellable)",
        "msg_export_cancelled": "Export cancelled",
        "msg_orientation_title": "Orientation",
        "msg_orientation_deprecated": "orientation_distribution=mixture is deprecated in GUI; fallback to normal.",
        "msg_meta_random": "dim={dim} shape={shape} mask_ratio={mask_ratio:.3f} seed={seed}",
        "msg_meta_invert": "dim={dim} shape={shape} mask_ratio={mask_ratio:.3f}",
        "value_mid": "mid",
        "value_optional": "(optional)",
        "default_prefix_random": "random_field",
        "default_prefix_invert": "image_field",
        "table_gb_mean": "Mean",
        "table_gb_std": "Std",
        "table_gb_weight": "Weight",
        "table_gb_note": "Note",
        "combo_orientation_random": "Random",
        "combo_orientation_normal": "Normal",
        "combo_sampler_normal": "Normal",
        "combo_sampler_mixture": "Mixture",
        "combo_theta_truncate": "Truncate",
        "combo_theta_clip": "Clip",
        "combo_grain_uniform": "Uniform",
        "combo_grain_normal": "Normal",
        "combo_grain_lognormal": "Lognormal",
        "combo_defect_uniform": "Uniform",
        "combo_defect_normal": "Normal",
        "combo_defect_lognormal": "Lognormal",
        "combo_defect_sphere": "Sphere",
        "combo_defect_ellipsoid": "Ellipsoid",
        "combo_defect_blob": "Blob",
        "combo_defect_disk": "Disk",
        "combo_defect_ellipse": "Ellipse",
        "combo_defect_policy_nan": "NaN",
        "combo_defect_policy_zero": "Zero",
        "combo_defect_policy_fill": "Fill",
        "combo_cbar_top_high": "Top = high",
        "combo_cbar_top_low": "Top = low",
        "error_theta_range": "theta_min must be less than theta_max",
        "error_gb_hard": "gb_mis_hard_max must be > 0",
        "error_defect_fraction": "defect_fraction must be in [0,1]",
        "error_shape_positive": "shape must be positive",
        "error_shape_z_positive": "shape_z must be positive",
        "error_grain_std": "grain_size_std must be <= grain_size_mean",
        "error_grain_too_large": "Grain size too large; may leave only a few grains",
        "error_mixture_format": "mixture format invalid",
        "error_mixture_required": "mixture input required",
        "tip_dim_2d": "2D orientation field",
        "tip_dim_3d": "3D orientation field",
        "tip_shape": "Grid size, must be positive integers",
        "tip_grain_mean": "Mean grain size (px/voxel), affects grain_id topology.",
        "tip_grain_std": "Grain size dispersion (std).",
        "tip_grain_dist": "Grain size distribution: uniform/normal/lognormal.",
        "tip_shape_z": "Used only in 3D; hidden in 2D.",
        "tip_orientation_dist": "Absolute orientation (θ) distribution",
        "tip_mean_theta": "Absolute orientation θ mean (deg)",
        "tip_orientation_std": "Absolute orientation θ std (deg)",
        "tip_theta_range": "Absolute orientation θ range (deg)",
        "tip_theta_policy": "Absolute orientation θ boundary policy",
        "tip_gb_hard": "GB misorientation Δθ hard limit",
        "tip_gb_target": "GB misorientation Δθ target (P95)",
        "tip_gb_sampler": "GB misorientation (Δθ) sampler",
        "tip_texture_mixture": "This string defines the GB misorientation Δθ mixture parameters (mean/std/weight), not absolute θ.",
        "tip_texture_table": "This table defines the GB misorientation Δθ mixture parameters (mean/std/weight), not absolute θ.",
        "tip_defect_enabled": "Enable defects/mask",
        "tip_defect_fraction": "Defect fraction [0,1]",
        "tip_defect_size_mean": "Defect size mean",
        "tip_defect_size_std": "Defect size std",
        "tip_defect_size_dist": "Defect size distribution",
        "tip_pbc_x": "Enable x periodic boundary",
        "tip_pbc_y": "Enable y periodic boundary",
        "tip_pbc_z": "Enable z periodic boundary",
        "tip_outdir": "Output directory",
        "tip_prefix": "Output filename prefix",
        "tip_seed": "Seed (optional); blank for auto",
        "tip_axis": "Select slice axis",
        "tip_slice_slider": "Slice index",
        "tip_slice_spin": "Slice index",
        "tip_mid": "Jump to middle slice",
        "tip_export_dir": "Export directory",
        "tip_export_npz": "Export NPZ",
        "tip_export_json": "Export JSON",
        "tip_export_csv": "Export CSV",
        "tip_export_h5": "Export HDF5",
        "tip_csv_axis": "3D CSV axis",
        "tip_csv_index": "Slice index or mid",
        "tip_field_image": "Input field image",
        "tip_cbar_image": "Input colorbar image",
        "tip_cbar_min": "Colorbar min (deg)",
        "tip_cbar_max": "Colorbar max (deg)",
        "tip_cbar_direction": "Colorbar direction (top_high / top_low)",
        "tip_defect_policy": "Defect region policy",
        "tip_color_tolerance": "Color tolerance",
        "tip_defect_threshold": "Defect threshold",
    },
}


def t(lang: str, key: str) -> str:
    return TEXT.get(lang, TEXT["zh"]).get(key, key)


def tf(lang: str, key: str, **kwargs: Any) -> str:
    return t(lang, key).format(**kwargs)


def is_dark_theme(palette: QtGui.QPalette) -> bool:
    color = palette.color(QtGui.QPalette.ColorRole.Window)
    luminance = 0.2126 * color.red() + 0.7152 * color.green() + 0.0722 * color.blue()
    return luminance < 128


@dataclass
class RandomRunState:
    config: Optional[RandomFieldConfig] = None
    npz_path: Optional[Path] = None
    preview_dir: Optional[Path] = None
    meta: Optional[Dict[str, Any]] = None
    outdir: Optional[Path] = None
    prefix: str = "random_field"


@dataclass
class InvertRunState:
    config: Optional[FromImagesConfig] = None
    npz_path: Optional[Path] = None
    preview_dir: Optional[Path] = None
    meta: Optional[Dict[str, Any]] = None
    outdir: Optional[Path] = None
    prefix: str = "image_field"


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str)


class Worker(QtCore.QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._cancel = False

    def run(self) -> None:
        try:
            if self._cancel:
                return
            result = self.fn(*self.args, **self.kwargs)
            if self._cancel:
                return
        except Exception as exc:
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(result)

    def request_cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel


class RenderProgressDialog(QtWidgets.QDialog):
    def __init__(
        self,
        message: str,
        cancel_fn: Callable[[], None],
        lang: str,
        title: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._cancelled = False
        self._cancel_fn = cancel_fn
        self.setWindowTitle(title or t(lang, "dialog_render_title"))
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        label = QtWidgets.QLabel(message)
        label.setWordWrap(True)
        stop_btn = QtWidgets.QPushButton(t(lang, "btn_stop"))
        stop_btn.clicked.connect(self._on_cancel)

        layout.addWidget(label)
        layout.addWidget(stop_btn)

    def _on_cancel(self) -> None:
        self._cancelled = True
        self._cancel_fn()

    def allow_close(self) -> None:
        self._cancelled = True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._cancelled:
            event.accept()
        else:
            event.ignore()


class LogEmitter(QtCore.QObject):
    message = QtCore.Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter) -> None:
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.emitter.message.emit(msg)


class PageBase(QtWidgets.QWidget):
    def __init__(self, main: "MainWindow") -> None:
        super().__init__()
        self.main = main
        self._translations: list[Callable[[str], None]] = []

    def set_enabled(self, enabled: bool) -> None:
        for widget in self.findChildren(QtWidgets.QWidget):
            if widget is self:
                continue
            widget.setEnabled(enabled)

    def _register_text(self, widget: QtWidgets.QWidget, key: str, *, attr: str = "setText") -> None:
        def apply(lang: str) -> None:
            getattr(widget, attr)(t(lang, key))

        self._translations.append(apply)
        apply(self.main.lang)

    def _register_tooltip(self, widget: QtWidgets.QWidget, key: str) -> None:
        def apply(lang: str) -> None:
            widget.setToolTip(t(lang, key))

        self._translations.append(apply)
        apply(self.main.lang)

    def _make_label(self, key: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel()
        self._register_text(label, key)
        return label

    def _register_combo_items(self, combo: QtWidgets.QComboBox, items: list[tuple[str, str]]) -> None:
        for value, key in items:
            combo.addItem(t(self.main.lang, key), value)

        def apply(lang: str) -> None:
            for index, (_, item_key) in enumerate(items):
                combo.setItemText(index, t(lang, item_key))

        self._translations.append(apply)

    def apply_language(self, lang: str) -> None:
        for updater in self._translations:
            updater(lang)

    def _build_nav(
        self,
        *,
        back_cb: Optional[Callable[[], None]] = None,
        home_cb: Optional[Callable[[], None]] = None,
        run_cb: Optional[Callable[[], None]] = None,
        next_cb: Optional[Callable[[], None]] = None,
        show_back: bool = True,
        show_home: bool = True,
        show_run: bool = False,
        show_next: bool = True,
    ) -> QtWidgets.QWidget:
        nav = QtWidgets.QHBoxLayout()
        if show_back:
            back = QtWidgets.QPushButton()
            self._register_text(back, "btn_back")
            if back_cb:
                back.clicked.connect(back_cb)
            else:
                back.setEnabled(False)
            nav.addWidget(back)
        if show_home:
            home = QtWidgets.QPushButton()
            self._register_text(home, "btn_home")
            if home_cb:
                home.clicked.connect(home_cb)
            else:
                home.setEnabled(False)
            nav.addWidget(home)
        nav.addStretch()
        if show_run:
            run = QtWidgets.QPushButton()
            self._register_text(run, "btn_run")
            if run_cb:
                run.clicked.connect(run_cb)
            else:
                run.setEnabled(False)
            nav.addWidget(run)
        if show_next:
            next_btn = QtWidgets.QPushButton()
            self._register_text(next_btn, "btn_next")
            if next_cb:
                next_btn.clicked.connect(next_cb)
            else:
                next_btn.setEnabled(False)
            nav.addWidget(next_btn)
        nav_widget = QtWidgets.QWidget()
        nav_widget.setLayout(nav)
        return nav_widget


class HomePage(PageBase):
    def __init__(self, main: "MainWindow", lang: str) -> None:
        super().__init__(main)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QtWidgets.QLabel()
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        subtitle = QtWidgets.QLabel()
        subtitle.setStyleSheet("color: #666;")
        self._register_text(title, "welcome_title")
        self._register_text(subtitle, "welcome_subtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.setSpacing(16)

        random_card, random_button, random_title, random_desc = self._build_card(
            lambda: main.show_page("random_config")
        )
        self._register_text(random_title, "welcome_random")
        self._register_text(random_desc, "welcome_random_desc")
        images_card, images_button, images_title, images_desc = self._build_card(
            lambda: main.show_page("invert_config")
        )
        self._register_text(images_title, "welcome_images")
        self._register_text(images_desc, "welcome_images_desc")
        self._apply_home_theme_styles(
            [random_card, images_card],
            [random_button, images_button],
        )
        buttons_layout.addWidget(random_card)
        buttons_layout.addWidget(images_card)
        layout.addLayout(buttons_layout)

    def _build_card(
        self, callback: Callable[[], None]
    ) -> tuple[QtWidgets.QFrame, QtWidgets.QPushButton, QtWidgets.QLabel, QtWidgets.QLabel]:
        card = QtWidgets.QFrame()
        card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        card.setAutoFillBackground(True)
        layout = QtWidgets.QVBoxLayout(card)
        label = QtWidgets.QLabel()
        label.setStyleSheet("font-size: 18px; font-weight: 600;")
        desc = QtWidgets.QLabel()
        desc.setWordWrap(True)
        button = QtWidgets.QPushButton()
        self._register_text(button, "btn_start")
        button.clicked.connect(callback)
        layout.addWidget(label)
        layout.addWidget(desc)
        layout.addStretch()
        layout.addWidget(button)
        return card, button, label, desc

    def _apply_home_theme_styles(
        self,
        cards: list[QtWidgets.QFrame],
        buttons: list[QtWidgets.QPushButton],
    ) -> None:
        if is_dark_theme(self.palette()):
            card_style = """
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #3a3a3a;
                border-radius: 10px;
                padding: 12px;
            }
            QLabel { color: #eaeaea; }
            """
            button_style = """
            QPushButton {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:pressed { background-color: #2f2f2f; }
            """
        else:
            card_style = """
            QFrame {
                background-color: #f5f5f5;
                border-radius: 8px;
                padding: 12px;
            }
            """
            button_style = """
            QPushButton {
                color: palette(ButtonText);
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: palette(Highlight);
                color: palette(HighlightedText);
            }
            QPushButton:pressed {
                background-color: palette(Highlight);
                color: palette(HighlightedText);
            }
            """

        for card in cards:
            card.setStyleSheet(card_style)
        for button in buttons:
            button.setStyleSheet(button_style)


class RandomConfigPage(PageBase):
    def __init__(self, main: "MainWindow", lang: str) -> None:
        super().__init__(main)
        self.lang = lang
        self._mixture_warned = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content.setMinimumWidth(760)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)
        content_layout.addWidget(self._build_geometry_group())
        content_layout.addWidget(self._build_topology_group())
        content_layout.addWidget(self._build_absolute_group())
        content_layout.addWidget(self._build_gb_group())
        content_layout.addWidget(self._build_defect_group())
        content_layout.addWidget(self._build_pbc_group())
        content_layout.addWidget(self._build_output_group())
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        nav = self._build_nav(
            home_cb=lambda: self.main.show_page("home"),
            next_cb=self._go_next,
            show_back=False,
            show_run=False,
            show_next=True,
        )
        layout.addWidget(nav)

        self._update_dimension_fields()
        self._update_defect_fields()
        self._update_texture_fields()
        self._translations.append(lambda lang: self._update_defect_shape_texts(lang))

    def _build_geometry_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_geometry", attr="setTitle")
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.dim_2d = QtWidgets.QRadioButton("2D")
        self.dim_3d = QtWidgets.QRadioButton("3D")
        self.dim_2d.setChecked(True)
        self._register_tooltip(self.dim_2d, "tip_dim_2d")
        self._register_tooltip(self.dim_3d, "tip_dim_3d")
        dim_layout = QtWidgets.QHBoxLayout()
        dim_layout.setSpacing(8)
        dim_layout.addWidget(self.dim_2d)
        dim_layout.addWidget(self.dim_3d)
        self.dim_2d.toggled.connect(self._update_dimension_fields)

        dim_widget = QtWidgets.QWidget()
        dim_widget.setLayout(dim_layout)
        layout.addRow(self._make_label("label_dim"), dim_widget)

        self.shape_x = QtWidgets.QSpinBox()
        self.shape_y = QtWidgets.QSpinBox()
        self.shape_z = QtWidgets.QSpinBox()
        for spin in (self.shape_x, self.shape_y):
            spin.setRange(1, 4096)
            spin.setValue(256)
            self._register_tooltip(spin, "tip_shape")
        self.shape_z.setRange(1, 4096)
        self.shape_z.setValue(256)
        self._register_tooltip(self.shape_z, "tip_shape_z")
        layout.addRow(self._make_label("label_shape_x"), self.shape_x)
        layout.addRow(self._make_label("label_shape_y"), self.shape_y)
        layout.addRow(self._make_label("label_shape_z"), self.shape_z)

        self.grain_size_mean = QtWidgets.QDoubleSpinBox()
        self.grain_size_mean.setRange(1.0, 512.0)
        self.grain_size_mean.setDecimals(1)
        self.grain_size_mean.setValue(32.0)
        self._register_tooltip(self.grain_size_mean, "tip_grain_mean")

        self.grain_size_std = QtWidgets.QDoubleSpinBox()
        self.grain_size_std.setRange(0.0, self.grain_size_mean.value())
        self.grain_size_std.setDecimals(1)
        self.grain_size_std.setValue(6.0)
        self._register_tooltip(self.grain_size_std, "tip_grain_std")

        self.grain_size_distribution = QtWidgets.QComboBox()
        self._register_combo_items(
            self.grain_size_distribution,
            [
                ("uniform", "combo_grain_uniform"),
                ("normal", "combo_grain_normal"),
                ("lognormal", "combo_grain_lognormal"),
            ],
        )
        self.grain_size_distribution.setCurrentIndex(self.grain_size_distribution.findData("normal"))
        self._register_tooltip(self.grain_size_distribution, "tip_grain_dist")

        self.shape_x.valueChanged.connect(self._update_grain_size_limits)
        self.shape_y.valueChanged.connect(self._update_grain_size_limits)
        self.shape_z.valueChanged.connect(self._update_grain_size_limits)
        self.grain_size_mean.valueChanged.connect(self._update_grain_size_limits)
        return group

    def _build_topology_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_topology", attr="setTitle")
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        layout.addRow(self._make_label("label_grain_mean"), self.grain_size_mean)
        layout.addRow(self._make_label("label_grain_std"), self.grain_size_std)
        layout.addRow(self._make_label("label_grain_dist"), self.grain_size_distribution)
        return group

    def _build_absolute_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_absolute", attr="setTitle")
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.orientation_dist = QtWidgets.QComboBox()
        self._register_combo_items(
            self.orientation_dist,
            [
                ("random", "combo_orientation_random"),
                ("normal", "combo_orientation_normal"),
            ],
        )
        self.orientation_dist.currentTextChanged.connect(self._update_texture_fields)
        self._register_tooltip(self.orientation_dist, "tip_orientation_dist")
        self.mean_theta = QtWidgets.QDoubleSpinBox()
        self.mean_theta.setRange(-180.0, 180.0)
        self.mean_theta.setValue(0.0)
        self._register_tooltip(self.mean_theta, "tip_mean_theta")
        self.orientation_std = QtWidgets.QDoubleSpinBox()
        self.orientation_std.setRange(0.1, 90.0)
        self.orientation_std.setValue(10.0)
        self._register_tooltip(self.orientation_std, "tip_orientation_std")

        self.theta_min = QtWidgets.QDoubleSpinBox()
        self.theta_max = QtWidgets.QDoubleSpinBox()
        for spin in (self.theta_min, self.theta_max):
            spin.setRange(-360.0, 360.0)
            spin.setDecimals(2)
            self._register_tooltip(spin, "tip_theta_range")
        self.theta_min.setValue(-45.0)
        self.theta_max.setValue(45.0)

        self.theta_boundary_policy = QtWidgets.QComboBox()
        self._register_combo_items(
            self.theta_boundary_policy,
            [
                ("truncate", "combo_theta_truncate"),
                ("clip", "combo_theta_clip"),
            ],
        )
        self._register_tooltip(self.theta_boundary_policy, "tip_theta_policy")

        layout.addRow(self._make_label("label_theta_min"), self.theta_min)
        layout.addRow(self._make_label("label_theta_max"), self.theta_max)
        layout.addRow(self._make_label("label_theta_policy"), self.theta_boundary_policy)
        layout.addRow(self._make_label("label_orientation_dist"), self.orientation_dist)
        layout.addRow(self._make_label("label_mean_theta"), self.mean_theta)
        layout.addRow(self._make_label("label_orientation_std"), self.orientation_std)
        return group

    def _build_gb_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_gb", attr="setTitle")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        self.gb_hard = QtWidgets.QDoubleSpinBox()
        self.gb_hard.setRange(0.1, 180.0)
        self.gb_hard.setValue(25.0)
        self.gb_target = QtWidgets.QDoubleSpinBox()
        self.gb_target.setRange(0.0, 180.0)
        self.gb_target.setValue(20.0)
        self.gb_target.setSpecialValueText(t(self.main.lang, "value_optional"))
        self._translations.append(lambda lang: self.gb_target.setSpecialValueText(t(lang, "value_optional")))
        self.gb_target.setValue(0.0)
        self._register_tooltip(self.gb_hard, "tip_gb_hard")
        self._register_tooltip(self.gb_target, "tip_gb_target")

        self.gb_sampler = QtWidgets.QComboBox()
        self._register_combo_items(
            self.gb_sampler,
            [
                ("normal", "combo_sampler_normal"),
                ("mixture", "combo_sampler_mixture"),
            ],
        )
        self.gb_sampler.currentTextChanged.connect(self._on_gb_sampler_changed)
        self._register_tooltip(self.gb_sampler, "tip_gb_sampler")

        form.addRow(self._make_label("label_gb_hard"), self.gb_hard)
        form.addRow(self._make_label("label_gb_target"), self.gb_target)
        form.addRow(self._make_label("label_gb_sampler"), self.gb_sampler)
        layout.addLayout(form)

        self.texture_group = QtWidgets.QGroupBox()
        self._register_text(self.texture_group, "group_texture_advanced", attr="setTitle")
        texture_layout = QtWidgets.QVBoxLayout(self.texture_group)
        texture_layout.setContentsMargins(10, 10, 10, 10)
        texture_layout.setSpacing(8)
        self.texture_mixture_edit = QtWidgets.QLineEdit()
        self._register_tooltip(self.texture_mixture_edit, "tip_texture_mixture")
        self.texture_mixture_edit.setVisible(False)
        texture_layout.addWidget(self.texture_mixture_edit)
        self.texture_table = QtWidgets.QTableWidget(0, 4)
        self.texture_table.setHorizontalHeaderLabels(
            [
                t(self.main.lang, "table_gb_mean"),
                t(self.main.lang, "table_gb_std"),
                t(self.main.lang, "table_gb_weight"),
                t(self.main.lang, "table_gb_note"),
            ]
        )
        self.texture_table.horizontalHeader().setStretchLastSection(True)
        self._register_tooltip(self.texture_table, "tip_texture_table")
        self._translations.append(
            lambda lang: self.texture_table.setHorizontalHeaderLabels(
                [
                    t(lang, "table_gb_mean"),
                    t(lang, "table_gb_std"),
                    t(lang, "table_gb_weight"),
                    t(lang, "table_gb_note"),
                ]
            )
        )
        texture_layout.addWidget(self.texture_table)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.setSpacing(8)
        self.btn_add_row = QtWidgets.QPushButton()
        self.btn_remove_row = QtWidgets.QPushButton()
        self.btn_normalize = QtWidgets.QPushButton()
        self._register_text(self.btn_add_row, "btn_add")
        self._register_text(self.btn_remove_row, "btn_remove")
        self._register_text(self.btn_normalize, "btn_normalize")
        self.btn_add_row.clicked.connect(self._add_texture_row)
        self.btn_remove_row.clicked.connect(self._remove_texture_row)
        self.btn_normalize.clicked.connect(self._normalize_texture_weights)
        buttons_layout.addWidget(self.btn_add_row)
        buttons_layout.addWidget(self.btn_remove_row)
        buttons_layout.addWidget(self.btn_normalize)
        buttons_layout.addStretch()
        texture_layout.addLayout(buttons_layout)

        layout.addWidget(self.texture_group)
        return group

    def _build_defect_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_defect", attr="setTitle")
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.defect_enabled = QtWidgets.QCheckBox()
        self._register_text(self.defect_enabled, "label_defect_enabled")
        self.defect_enabled.toggled.connect(self._update_defect_fields)
        self._register_tooltip(self.defect_enabled, "tip_defect_enabled")
        self.defect_fraction = QtWidgets.QDoubleSpinBox()
        self.defect_fraction.setRange(0.0, 1.0)
        self.defect_fraction.setDecimals(3)
        self.defect_fraction.setValue(0.0)
        self._register_tooltip(self.defect_fraction, "tip_defect_fraction")
        self.defect_shape = QtWidgets.QComboBox()
        self.defect_size_mean = QtWidgets.QDoubleSpinBox()
        self.defect_size_mean.setRange(0.1, 256.0)
        self.defect_size_mean.setValue(8.0)
        self._register_tooltip(self.defect_size_mean, "tip_defect_size_mean")
        self.defect_size_std = QtWidgets.QDoubleSpinBox()
        self.defect_size_std.setRange(0.1, 128.0)
        self.defect_size_std.setValue(2.0)
        self._register_tooltip(self.defect_size_std, "tip_defect_size_std")
        self.defect_size_dist = QtWidgets.QComboBox()
        self._register_combo_items(
            self.defect_size_dist,
            [
                ("normal", "combo_defect_normal"),
                ("uniform", "combo_defect_uniform"),
                ("lognormal", "combo_defect_lognormal"),
            ],
        )
        self._register_tooltip(self.defect_size_dist, "tip_defect_size_dist")

        layout.addRow(self.defect_enabled)
        layout.addRow(self._make_label("label_defect_fraction"), self.defect_fraction)
        layout.addRow(self._make_label("label_defect_shape"), self.defect_shape)
        layout.addRow(self._make_label("label_defect_size_mean"), self.defect_size_mean)
        layout.addRow(self._make_label("label_defect_size_std"), self.defect_size_std)
        layout.addRow(self._make_label("label_defect_size_dist"), self.defect_size_dist)
        return group

    def _build_pbc_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_pbc", attr="setTitle")
        layout = QtWidgets.QHBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self._make_label("label_periodic"))
        self.pbc_x = QtWidgets.QCheckBox("x")
        self.pbc_y = QtWidgets.QCheckBox("y")
        self.pbc_z = QtWidgets.QCheckBox("z")
        self._register_tooltip(self.pbc_x, "tip_pbc_x")
        self._register_tooltip(self.pbc_y, "tip_pbc_y")
        self._register_tooltip(self.pbc_z, "tip_pbc_z")
        layout.addWidget(self.pbc_x)
        layout.addWidget(self.pbc_y)
        layout.addWidget(self.pbc_z)
        layout.addStretch()
        return group

    def _build_output_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_output", attr="setTitle")
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.outdir = QtWidgets.QLineEdit(str(OUTPUT_DIR))
        self._prefix_default = t(self.main.lang, "default_prefix_random")
        self.prefix = QtWidgets.QLineEdit(self._prefix_default)
        self.seed = QtWidgets.QLineEdit("")
        self._register_tooltip(self.outdir, "tip_outdir")
        self._register_tooltip(self.prefix, "tip_prefix")
        self._register_tooltip(self.seed, "tip_seed")

        def update_prefix_default(lang: str) -> None:
            new_default = t(lang, "default_prefix_random")
            if self.prefix.text() == self._prefix_default:
                self.prefix.setText(new_default)
            self._prefix_default = new_default

        self._translations.append(update_prefix_default)

        browse_btn = QtWidgets.QPushButton()
        self._register_text(browse_btn, "btn_browse")
        browse_btn.clicked.connect(self._choose_outdir)
        out_layout = QtWidgets.QHBoxLayout()
        out_layout.setSpacing(8)
        out_layout.addWidget(self.outdir)
        out_layout.addWidget(browse_btn)
        out_widget = QtWidgets.QWidget()
        out_widget.setLayout(out_layout)

        layout.addRow(self._make_label("label_outdir"), out_widget)
        layout.addRow(self._make_label("label_prefix"), self.prefix)
        layout.addRow(self._make_label("label_seed"), self.seed)
        return group

    def _choose_outdir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, t(self.main.lang, "label_outdir"))
        if path:
            self.outdir.setText(path)

    def _update_dimension_fields(self) -> None:
        is_3d = self.dim_3d.isChecked()
        self.shape_z.setVisible(is_3d)
        self.pbc_z.setVisible(is_3d)
        current_shape = self.defect_shape.currentData()
        if is_3d:
            self.defect_shape.clear()
            self.defect_shape.addItem(t(self.main.lang, "combo_defect_sphere"), "sphere")
            self.defect_shape.addItem(t(self.main.lang, "combo_defect_ellipsoid"), "ellipsoid")
            self.defect_shape.addItem(t(self.main.lang, "combo_defect_blob"), "blob")
        else:
            self.defect_shape.clear()
            self.defect_shape.addItem(t(self.main.lang, "combo_defect_disk"), "disk")
            self.defect_shape.addItem(t(self.main.lang, "combo_defect_ellipse"), "ellipse")
            self.defect_shape.addItem(t(self.main.lang, "combo_defect_blob"), "blob")
        if current_shape:
            index = self.defect_shape.findData(current_shape)
            if index >= 0:
                self.defect_shape.setCurrentIndex(index)
        self._update_grain_size_limits()

    def _update_defect_shape_texts(self, lang: str) -> None:
        current_shape = self.defect_shape.currentData()
        for index in range(self.defect_shape.count()):
            value = self.defect_shape.itemData(index)
            if value == "sphere":
                key = "combo_defect_sphere"
            elif value == "ellipsoid":
                key = "combo_defect_ellipsoid"
            elif value == "blob":
                key = "combo_defect_blob"
            elif value == "disk":
                key = "combo_defect_disk"
            else:
                key = "combo_defect_ellipse"
            self.defect_shape.setItemText(index, t(lang, key))
        if current_shape:
            index = self.defect_shape.findData(current_shape)
            if index >= 0:
                self.defect_shape.setCurrentIndex(index)

    def _update_grain_size_limits(self) -> None:
        if not hasattr(self, "grain_size_mean"):
            return
        min_dim = min(self.shape_x.value(), self.shape_y.value())
        if self.dim_3d.isChecked():
            min_dim = min(min_dim, self.shape_z.value())
        max_mean = max(1.0, min_dim / 2.0)
        current_mean = self.grain_size_mean.value()
        self.grain_size_mean.setMaximum(max(1.0, max_mean))
        if current_mean > self.grain_size_mean.maximum():
            self.grain_size_mean.setValue(self.grain_size_mean.maximum())
        self.grain_size_std.setMaximum(self.grain_size_mean.value())
        if self.grain_size_std.value() > self.grain_size_mean.value():
            self.grain_size_std.setValue(self.grain_size_mean.value())

    def _update_defect_fields(self) -> None:
        enabled = self.defect_enabled.isChecked()
        for widget in (
            self.defect_fraction,
            self.defect_shape,
            self.defect_size_mean,
            self.defect_size_std,
            self.defect_size_dist,
        ):
            widget.setEnabled(enabled)

    def _update_texture_fields(self) -> None:
        dist = self.orientation_dist.currentData() or self.orientation_dist.currentText()
        self.mean_theta.setEnabled(dist == "normal")
        self.orientation_std.setEnabled(dist == "normal")
        gb_sampler = (
            self.gb_sampler.currentData() or self.gb_sampler.currentText()
            if hasattr(self, "gb_sampler")
            else "normal"
        )
        use_table = gb_sampler == "mixture"
        self.texture_group.setVisible(use_table)
        if hasattr(self, "texture_mixture_edit"):
            self.texture_mixture_edit.setVisible(use_table)

    def _add_texture_row(self) -> None:
        row = self.texture_table.rowCount()
        self.texture_table.insertRow(row)
        for col in range(3):
            item = QtWidgets.QTableWidgetItem("0")
            self.texture_table.setItem(row, col, item)
        self.texture_table.setItem(row, 3, QtWidgets.QTableWidgetItem(""))

    def _remove_texture_row(self) -> None:
        row = self.texture_table.currentRow()
        if row >= 0:
            self.texture_table.removeRow(row)

    def _normalize_texture_weights(self) -> None:
        weights = []
        for row in range(self.texture_table.rowCount()):
            item = self.texture_table.item(row, 2)
            if item is None:
                continue
            try:
                weight = float(item.text())
            except ValueError:
                weight = 0.0
            weights.append(weight)
        total = sum(weights)
        if total <= 0:
            return
        for row, weight in enumerate(weights):
            normalized = weight / total
            self.texture_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{normalized:.4f}"))

    def _collect_texture_mixture(self) -> tuple[list[tuple[float, float]], list[float]]:
        if (self.gb_sampler.currentData() or self.gb_sampler.currentText()) != "mixture":
            return [], []
        if hasattr(self, "texture_table"):
            components: list[tuple[float, float]] = []
            weights: list[float] = []
            for row in range(self.texture_table.rowCount()):
                mean_item = self.texture_table.item(row, 0)
                std_item = self.texture_table.item(row, 1)
                weight_item = self.texture_table.item(row, 2)
                if mean_item is None or std_item is None or weight_item is None:
                    continue
                try:
                    mean_val = float(mean_item.text())
                    std_val = float(std_item.text())
                    weight_val = float(weight_item.text())
                except ValueError as exc:
                    raise ValueError("mixture values must be numeric") from exc
                if std_val <= 0 or weight_val <= 0:
                    raise ValueError("mixture std/weight must be positive")
                components.append((mean_val, std_val))
                weights.append(weight_val)
            if components or weights:
                return components, weights
        if hasattr(self, "texture_mixture_edit"):
            raw = self.texture_mixture_edit.text().strip()
            if not raw:
                return [], []
            components = []
            weights = []
            for part in raw.split(";"):
                if not part.strip():
                    continue
                try:
                    mean_str, std_str, weight_str = [item.strip() for item in part.split(",")]
                    mean_val = float(mean_str)
                    std_val = float(std_str)
                    weight_val = float(weight_str)
                except ValueError as exc:
                    raise ValueError("mixture format invalid") from exc
                if std_val <= 0 or weight_val <= 0:
                    raise ValueError("mixture std/weight must be positive")
                components.append((mean_val, std_val))
                weights.append(weight_val)
            return components, weights
        return [], []

    def _go_next(self) -> None:
        error = self._validate()
        if error:
            QtWidgets.QMessageBox.warning(self, t(self.main.lang, "msg_validation_title"), error)
            return
        self.main.random_state.config = self._build_config()
        self.main.random_state.outdir = Path(self.outdir.text())
        self.main.random_state.prefix = self.prefix.text().strip() or t(self.main.lang, "default_prefix_random")
        self.main.show_page("random_visualize")

    def _validate(self) -> Optional[str]:
        if self.theta_min.value() >= self.theta_max.value():
            return t(self.main.lang, "error_theta_range")
        if self.gb_hard.value() <= 0:
            return t(self.main.lang, "error_gb_hard")
        if self.defect_fraction.value() < 0 or self.defect_fraction.value() > 1:
            return t(self.main.lang, "error_defect_fraction")
        if self.shape_x.value() <= 0 or self.shape_y.value() <= 0:
            return t(self.main.lang, "error_shape_positive")
        if self.dim_3d.isChecked() and self.shape_z.value() <= 0:
            return t(self.main.lang, "error_shape_z_positive")
        if self.grain_size_std.value() > self.grain_size_mean.value():
            return t(self.main.lang, "error_grain_std")
        min_dim = min(self.shape_x.value(), self.shape_y.value())
        if self.dim_3d.isChecked():
            min_dim = min(min_dim, self.shape_z.value())
        if self.grain_size_mean.value() > (min_dim / 2.0):
            return t(self.main.lang, "error_grain_too_large")
        if (self.gb_sampler.currentData() or self.gb_sampler.currentText()) == "mixture":
            try:
                components, weights = self._collect_texture_mixture()
            except ValueError:
                return t(self.main.lang, "error_mixture_format")
            if not components or not weights:
                return t(self.main.lang, "error_mixture_required")
        return None

    def _build_config(self) -> RandomFieldConfig:
        dim = 3 if self.dim_3d.isChecked() else 2
        shape = (
            self.shape_x.value(),
            self.shape_y.value(),
            self.shape_z.value() if dim == 3 else None,
        )
        shape_tuple = tuple(value for value in shape if value is not None)

        periodic = ""
        if self.pbc_x.isChecked():
            periodic += "x"
        if self.pbc_y.isChecked():
            periodic += "y"
        if dim == 3 and self.pbc_z.isChecked():
            periodic += "z"
        if not periodic:
            periodic = "none"

        texture_components: Optional[list[tuple[float, float]]] = None
        texture_weights: Optional[list[float]] = None
        dist = self.orientation_dist.currentData() or self.orientation_dist.currentText()
        gb_sampler = self.gb_sampler.currentData() or self.gb_sampler.currentText()
        if gb_sampler == "mixture":
            components, weights = self._collect_texture_mixture()
            if weights and sum(weights) > 0:
                total = sum(weights)
                weights = [weight / total for weight in weights]
            texture_components = components or None
            texture_weights = weights or None

        seed_value = None
        if self.seed.text().strip():
            seed_value = int(self.seed.text().strip())

        gb_target = self.gb_target.value() if self.gb_target.value() > 0 else None
        return RandomFieldConfig(
            dim=dim,
            shape=shape_tuple,
            grain_size_mean=self.grain_size_mean.value(),
            grain_size_distribution=self.grain_size_distribution.currentData()
            or self.grain_size_distribution.currentText(),
            grain_size_std=self.grain_size_std.value(),
            orientation_distribution=dist,
            mean_theta_deg=self.mean_theta.value(),
            orientation_std_deg=self.orientation_std.value(),
            orientation_format="matrix",
            theta_min=self.theta_min.value(),
            theta_max=self.theta_max.value(),
            theta_boundary_policy=self.theta_boundary_policy.currentData()
            or self.theta_boundary_policy.currentText(),
            texture_mixture_components=texture_components,
            texture_mixture_weights=texture_weights,
            gb_mis_hard_max=self.gb_hard.value(),
            gb_mis_p95=gb_target,
            gb_mis_sampler=gb_sampler,
            gb_mis_std=None,
            gb_alpha=0.7,
            defect_enabled=self.defect_enabled.isChecked(),
            defect_fraction=self.defect_fraction.value(),
            defect_size_mean=self.defect_size_mean.value(),
            defect_size_distribution=self.defect_size_dist.currentData() or self.defect_size_dist.currentText(),
            defect_size_std=self.defect_size_std.value(),
            defect_threshold=1.0,
            defect_shape=self.defect_shape.currentData() or self.defect_shape.currentText(),
            periodic=periodic,
            seed=seed_value,
            seed_entropy=None,
        )

    def sync_from_state(self) -> None:
        config = self.main.random_state.config
        if config is None:
            return
        dist = config.orientation_distribution
        if dist == "mixture":
            dist = "normal"
            self.main.random_state.config = replace(config, orientation_distribution=dist)
            if not self._mixture_warned:
                QtWidgets.QMessageBox.information(
                    self,
                    t(self.main.lang, "msg_orientation_title"),
                    t(self.main.lang, "msg_orientation_deprecated"),
                )
                self._mixture_warned = True
        if dist in {"random", "normal"}:
            index = self.orientation_dist.findData(dist)
            if index >= 0:
                self.orientation_dist.setCurrentIndex(index)
        if config.gb_mis_sampler in {"normal", "mixture"}:
            index = self.gb_sampler.findData(config.gb_mis_sampler)
            if index >= 0:
                self.gb_sampler.setCurrentIndex(index)
        if config.gb_mis_sampler == "mixture":
            self.texture_table.setRowCount(0)
            components = config.texture_mixture_components or []
            weights = config.texture_mixture_weights or []
            for idx, (mean_val, std_val) in enumerate(components):
                row = self.texture_table.rowCount()
                self.texture_table.insertRow(row)
                weight_val = weights[idx] if idx < len(weights) else 1.0
                self.texture_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(mean_val)))
                self.texture_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(std_val)))
                self.texture_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(weight_val)))
                self.texture_table.setItem(row, 3, QtWidgets.QTableWidgetItem(""))
        self._update_texture_fields()

    def _on_gb_sampler_changed(self, value: str) -> None:
        self._update_texture_fields()
        if hasattr(self, "texture_mixture_edit"):
            sampler = self.gb_sampler.currentData() or value
            self.texture_mixture_edit.setVisible(sampler == "mixture")
            if sampler != "mixture":
                self.texture_mixture_edit.clear()


class RandomVisualizePage(PageBase):
    def __init__(self, main: "MainWindow", lang: str) -> None:
        super().__init__(main)
        self.lang = lang
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.status_label = QtWidgets.QLabel()
        self._status_key = "status_ready"
        self._status_args: dict[str, Any] = {}
        self._translations.append(
            lambda current_lang: self.status_label.setText(
                tf(current_lang, self._status_key, **self._status_args)
            )
        )
        self.apply_language(self.main.lang)
        layout.addWidget(self.status_label)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(["x", "y", "z"])
        self._register_tooltip(self.axis_combo, "tip_axis")
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._register_tooltip(self.slice_slider, "tip_slice_slider")
        self.slice_spin = QtWidgets.QSpinBox()
        self._register_tooltip(self.slice_spin, "tip_slice_spin")
        self.btn_mid = QtWidgets.QPushButton()
        self._register_text(self.btn_mid, "btn_mid")
        self._register_tooltip(self.btn_mid, "tip_mid")

        self.axis_combo.currentTextChanged.connect(self._on_axis_changed)
        self.slice_slider.valueChanged.connect(self.slice_spin.setValue)
        self.slice_spin.valueChanged.connect(self.slice_slider.setValue)
        self.slice_slider.valueChanged.connect(self._schedule_slice_update)
        self.btn_mid.clicked.connect(self._set_mid_slice)

        controls.addWidget(self._make_label("label_axis"))
        controls.addWidget(self.axis_combo)
        controls.addWidget(self._make_label("label_slice"))
        controls.addWidget(self.slice_slider, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        controls.addWidget(self.slice_spin)
        controls.addWidget(self.btn_mid)
        layout.addLayout(controls)

        viewer_layout = QtWidgets.QHBoxLayout()
        viewer_layout.setSpacing(8)
        viewer_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        preview_side = 320
        self.field_view = QtWidgets.QLabel()
        self.mask_view = QtWidgets.QLabel()
        for view in (self.field_view, self.mask_view):
            view.setFixedSize(preview_side, preview_side)
            view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            view.setScaledContents(False)
            view.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        viewer_layout.addWidget(self.field_view)
        viewer_layout.addWidget(self.mask_view)
        viewer_container = QtWidgets.QWidget()
        viewer_container.setLayout(viewer_layout)
        viewer_container.setContentsMargins(0, 0, 0, 0)
        viewer_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(viewer_container)

        self.meta_label = QtWidgets.QLabel("-")
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)
        self._meta_cache: Optional[Dict[str, Any]] = None
        self._translations.append(lambda lang: self._refresh_meta_label(lang))
        self._meta_cache: Optional[Dict[str, Any]] = None
        self._translations.append(lambda lang: self._refresh_meta_label(lang))

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(6)
        back = QtWidgets.QPushButton()
        self._register_text(back, "btn_back")
        back.clicked.connect(lambda: self.main.show_page("random_config"))
        home = QtWidgets.QPushButton()
        self._register_text(home, "btn_home")
        home.clicked.connect(lambda: self.main.show_page("home"))
        run = QtWidgets.QPushButton()
        self._register_text(run, "btn_run")
        run.clicked.connect(self._run_generation)
        next_btn = QtWidgets.QPushButton()
        self._register_text(next_btn, "btn_next")
        next_btn.clicked.connect(lambda: self.main.show_page("random_export"))
        nav.addWidget(back)
        nav.addWidget(home)
        nav.addStretch()
        nav.addWidget(run)
        nav.addWidget(next_btn)
        nav_widget = QtWidgets.QWidget()
        nav_widget.setLayout(nav)
        layout.addWidget(nav_widget)

        self._slice_timer = QtCore.QTimer(self)
        self._slice_timer.setSingleShot(True)
        self._slice_timer.timeout.connect(self._update_slice)

        self._update_slice_controls(enabled=False)
        self._cache_max_per_axis = 64
        self._slice_cache: dict[tuple[str, int], tuple[Path, Path, int]] = {}
        self._access_counter = 0
        self._active_dialog: Optional[RenderProgressDialog] = None
        self._active_worker: Optional[Worker] = None
        self._cancel_requested = False
        self._render_inflight = False
        self._pending_request: Optional[tuple[str, int]] = None
        self._active_req_id = 0
        self._cancelled_req_ids: set[int] = set()

    def _run_generation(self) -> None:
        config = self.main.random_state.config
        if config is None:
            QtWidgets.QMessageBox.warning(
                self,
                t(self.main.lang, "msg_missing_title"),
                t(self.main.lang, "msg_missing_config"),
            )
            return
        self._set_status("status_running")
        self._cancel_requested = False
        self._show_render_dialog(t(self.main.lang, "msg_rendering_preview"))
        self.set_enabled(False)

        def task() -> Dict[str, Any]:
            field = generate_random_field(config)
            mask_ratio = float(np.mean(field["mask"].astype(bool)))
            meta = build_meta(config, "random", field.get("_seed_meta", {}))
            gb_stats = field.get("gb_stats", {})
            meta.update(
                {
                    "mask_ratio": mask_ratio,
                    "gb_max": float(gb_stats.get("gb_mis_max", 0.0)),
                    "gb_hard_ok": float(gb_stats.get("gb_mis_max", 0.0)) <= float(config.gb_mis_hard_max),
                }
            )
            field["meta"] = meta

            outdir = self.main.random_state.outdir or OUTPUT_DIR
            outdir.mkdir(parents=True, exist_ok=True)
            prefix = self.main.random_state.prefix
            output_path = outdir / f"{prefix}.npz"
            output_path = write_npz(field, output_path)

            preview_dir = OUTPUT_DIR / "preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            vmin, vmax = resolve_global_vmin_vmax(field)
            field_path = preview_dir / f"{output_path.stem}_field.png"
            mask_path = preview_dir / f"{output_path.stem}_mask.png"
            if config.dim == 2:
                theta_2d, mask_2d = compute_slice_arrays_from_npz(output_path, "z", 0)
            else:
                mid_index = config.shape[2] // 2
                theta_2d, mask_2d = compute_slice_arrays_from_npz(output_path, "z", mid_index)
            render_field_and_mask_png(
                theta_2d,
                mask_2d,
                field_path,
                mask_path,
                vmin,
                vmax,
                PREVIEW_STYLE_CFG,
            )
            return {
                "npz": output_path,
                "preview_dir": preview_dir,
                "field_path": field_path,
                "mask_path": mask_path,
                "meta": meta,
            }

        worker = Worker(task)
        self._active_worker = worker
        worker.signals.finished.connect(self._on_generated)
        worker.signals.error.connect(self._on_error)
        self.main.thread_pool.start(worker)

    def _on_generated(self, payload: Dict[str, Any]) -> None:
        if self._cancel_requested:
            self._finish_render_dialog()
            return
        self.main.random_state.npz_path = payload["npz"]
        self.main.random_state.preview_dir = payload["preview_dir"]
        self.main.random_state.meta = payload["meta"]
        self._load_preview(payload["field_path"], payload["mask_path"])
        self._update_meta(payload["meta"])
        self._set_status("status_done")
        self.set_enabled(True)
        self._finish_render_dialog()
        self._update_slice_controls(enabled=self.main.random_state.config and self.main.random_state.config.dim == 3)

    def _on_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, t(self.main.lang, "msg_error_title"), message)
        self._set_status("status_error")
        self.set_enabled(True)
        self._finish_render_dialog()

    def _set_status(self, key: str, **kwargs: Any) -> None:
        self._status_key = key
        self._status_args = kwargs
        self.status_label.setText(tf(self.main.lang, key, **kwargs))

    def _update_meta(self, meta: Dict[str, Any]) -> None:
        self._meta_cache = meta
        self._refresh_meta_label(self.main.lang)

    def _refresh_meta_label(self, lang: str) -> None:
        if not self._meta_cache:
            return
        self.meta_label.setText(
            tf(
                lang,
                "msg_meta_random",
                dim=self._meta_cache.get("dim"),
                shape=self._meta_cache.get("shape"),
                mask_ratio=self._meta_cache.get("mask_ratio"),
                seed=self._meta_cache.get("seed_user"),
            )
        )

    def _load_preview(self, field_path: Path, mask_path: Path) -> None:
        self._load_image(self.field_view, field_path)
        self._load_image(self.mask_view, mask_path)

    @staticmethod
    def _load_image(label: QtWidgets.QLabel, path: Path) -> None:
        pixmap = QtGui.QPixmap(str(path))
        target = label.size()
        scaled = pixmap.scaled(
            target,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def _show_render_dialog(self, message: str) -> None:
        if self._active_dialog:
            self._finish_render_dialog()
        self._active_dialog = RenderProgressDialog(message, self._cancel_active_render, self.main.lang, None)
        self._active_dialog.show()

    def _finish_render_dialog(self) -> None:
        if self._active_dialog:
            self._active_dialog.allow_close()
            self._active_dialog.close()
            self._active_dialog = None
        self._active_worker = None
        self.set_enabled(True)

    def _cancel_active_render(self) -> None:
        self._cancel_requested = True
        if self._active_worker:
            self._active_worker.request_cancel()
        cancelled_id = self._active_req_id
        self._active_req_id += 1
        self._pending_request = None
        self._cancelled_req_ids.add(cancelled_id)
        self._render_inflight = False
        self._finish_render_dialog()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.main.random_state.preview_dir and self.main.random_state.npz_path:
            preview_dir = self.main.random_state.preview_dir
            prefix = self.main.random_state.npz_path.stem
            field_path = preview_dir / f"{prefix}_field.png"
            mask_path = preview_dir / f"{prefix}_mask.png"
            if field_path.exists():
                self._load_image(self.field_view, field_path)
            if mask_path.exists():
                self._load_image(self.mask_view, mask_path)

    def _update_slice_controls(self, enabled: bool) -> None:
        for widget in (self.axis_combo, self.slice_slider, self.slice_spin, self.btn_mid):
            widget.setEnabled(enabled)
        if not enabled:
            self.axis_combo.setVisible(False)
            self.slice_slider.setVisible(False)
            self.slice_spin.setVisible(False)
            self.btn_mid.setVisible(False)
        else:
            self.axis_combo.setVisible(True)
            self.slice_slider.setVisible(True)
            self.slice_spin.setVisible(True)
            self.btn_mid.setVisible(True)
            self._set_slider_range()

    def _set_slider_range(self) -> None:
        config = self.main.random_state.config
        if not config or config.dim != 3:
            return
        axis = self.axis_combo.currentText()
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        length = config.shape[axis_index]
        self.slice_slider.setRange(0, length - 1)
        self.slice_spin.setRange(0, length - 1)
        self.slice_slider.setValue(length // 2)

    def _schedule_slice_update(self) -> None:
        if self.main.random_state.npz_path is None:
            return
        self._slice_timer.start(150)

    def _set_mid_slice(self) -> None:
        self._set_slider_range()

    def _on_axis_changed(self) -> None:
        self._set_slider_range()
        self._schedule_slice_update()

    def _update_slice(self) -> None:
        if not self.main.random_state.npz_path:
            return
        axis = self.axis_combo.currentText()
        index = self.slice_slider.value()
        self._request_slice(axis, index)

    def _request_slice(self, axis: str, index: int) -> None:
        npz_path = self.main.random_state.npz_path
        if not npz_path:
            return
        if self._render_inflight:
            self._pending_request = (axis, index)
            return
        self._render_inflight = True
        self._pending_request = None
        self._active_req_id += 1
        req_id = self._active_req_id
        preview_dir = self.main.random_state.preview_dir or (OUTPUT_DIR / "preview")
        prefix = npz_path.stem

        cached = self._slice_cache.get((axis, index))
        if cached and cached[0].exists() and cached[1].exists():
            self._touch_cache((axis, index))
            self._load_preview(cached[0], cached[1])
            self._render_inflight = False
            return

        self._cancel_requested = False
        self._show_render_dialog(tf(self.main.lang, "msg_rendering_slice", axis=axis, index=index))
        # Keep status label clean; slice progress is shown in the modal dialog.

        self._submit_slice_render(
            npz_path,
            OUTPUT_DIR / "cache_slices" / npz_path.stem,
            prefix,
            axis,
            index,
            req_id,
            update_ui=True,
        )

    def _touch_cache(self, key: tuple[str, int]) -> None:
        cached = self._slice_cache.get(key)
        if not cached:
            return
        self._access_counter += 1
        self._slice_cache[key] = (cached[0], cached[1], self._access_counter)

    def _submit_slice_render(
        self,
        npz_path: Path,
        cache_dir: Path,
        prefix: str,
        axis: str,
        index: int,
        req_id: int,
        update_ui: bool,
    ) -> None:
        key = (axis, index)
        cache_dir.mkdir(parents=True, exist_ok=True)

        def task() -> Dict[str, Any]:
            slice_prefix = f"{prefix}_{axis}_{index:03d}"
            field_path, mask_path = render_slice_3d(npz_path, axis, slice_prefix, cache_dir, index)
            return {
                "key": key,
                "field": field_path,
                "mask": mask_path,
                "req_id": req_id,
                "update_ui": update_ui,
            }

        def on_finished(data: Dict[str, Any]) -> None:
            self._access_counter += 1
            self._slice_cache[data["key"]] = (data["field"], data["mask"], self._access_counter)
            self._evict_cache(axis)
            self._render_inflight = False
            if data["req_id"] in self._cancelled_req_ids:
                self._finish_render_dialog()
                self._maybe_start_pending()
                return
            if data["req_id"] != self._active_req_id:
                self._finish_render_dialog()
                self._maybe_start_pending()
                return
            if data["update_ui"] and not self._cancel_requested:
                self._load_preview(data["field"], data["mask"])
            self._finish_render_dialog()
            self._maybe_start_pending()

        worker = Worker(task)
        if update_ui:
            self._active_worker = worker
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(self._on_error)
        self.main.thread_pool.start(worker)

    def _evict_cache(self, axis: str) -> None:
        entries = [(key, value) for key, value in self._slice_cache.items() if key[0] == axis]
        if len(entries) <= self._cache_max_per_axis:
            return
        entries.sort(key=lambda item: item[1][2])
        excess = len(entries) - self._cache_max_per_axis
        for i in range(excess):
            key, value = entries[i]
            field_path, mask_path, _ = value
            if field_path.exists():
                field_path.unlink()
            if mask_path.exists():
                mask_path.unlink()
            self._slice_cache.pop(key, None)

    def _maybe_start_pending(self) -> None:
        if self._render_inflight or self._pending_request is None:
            return
        axis, index = self._pending_request
        self._pending_request = None
        self._request_slice(axis, index)


class RandomExportPage(PageBase):
    def __init__(self, main: "MainWindow", lang: str) -> None:
        super().__init__(main)
        self.lang = lang
        self._export_worker: Optional[Worker] = None
        self._export_dialog: Optional[RenderProgressDialog] = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_export", attr="setTitle")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.export_dir = QtWidgets.QLineEdit(str(OUTPUT_DIR / "exports"))
        self._register_tooltip(self.export_dir, "tip_export_dir")
        browse_btn = QtWidgets.QPushButton()
        self._register_text(browse_btn, "btn_browse")
        browse_btn.clicked.connect(self._choose_export_dir)
        out_layout = QtWidgets.QHBoxLayout()
        out_layout.setSpacing(6)
        out_layout.addWidget(self.export_dir)
        out_layout.addWidget(browse_btn)
        out_widget = QtWidgets.QWidget()
        out_widget.setLayout(out_layout)

        self.format_npz = QtWidgets.QCheckBox("NPZ")
        self.format_json = QtWidgets.QCheckBox("JSON")
        self.format_csv = QtWidgets.QCheckBox("CSV")
        self.format_h5 = QtWidgets.QCheckBox("HDF5")
        self._register_tooltip(self.format_npz, "tip_export_npz")
        self._register_tooltip(self.format_json, "tip_export_json")
        self._register_tooltip(self.format_csv, "tip_export_csv")
        self._register_tooltip(self.format_h5, "tip_export_h5")
        self.format_npz.setChecked(True)
        format_layout = QtWidgets.QHBoxLayout()
        format_layout.setSpacing(6)
        for widget in (self.format_npz, self.format_json, self.format_csv, self.format_h5):
            format_layout.addWidget(widget)
        format_widget = QtWidgets.QWidget()
        format_widget.setLayout(format_layout)

        self.csv_axis = QtWidgets.QComboBox()
        self.csv_axis.addItems(["x", "y", "z"])
        self._csv_mid_value = t(self.main.lang, "value_mid")
        self.csv_index = QtWidgets.QLineEdit(self._csv_mid_value)
        self._register_tooltip(self.csv_axis, "tip_csv_axis")
        self._register_tooltip(self.csv_index, "tip_csv_index")

        def update_csv_mid(lang: str) -> None:
            new_value = t(lang, "value_mid")
            if self.csv_index.text() == self._csv_mid_value:
                self.csv_index.setText(new_value)
            self._csv_mid_value = new_value

        self._translations.append(update_csv_mid)

        form.addRow(self._make_label("label_outdir"), out_widget)
        form.addRow(self._make_label("label_formats"), format_widget)
        form.addRow(self._make_label("label_csv_axis"), self.csv_axis)
        form.addRow(self._make_label("label_csv_index"), self.csv_index)

        layout.addWidget(group)
        layout.addStretch()

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(6)
        back = QtWidgets.QPushButton()
        self._register_text(back, "btn_back")
        back.clicked.connect(lambda: self.main.show_page("random_visualize"))
        home = QtWidgets.QPushButton()
        self._register_text(home, "btn_home")
        home.clicked.connect(lambda: self.main.show_page("home"))
        export_btn = QtWidgets.QPushButton()
        self._register_text(export_btn, "btn_export")
        export_btn.clicked.connect(self._export)
        nav.addWidget(back)
        nav.addWidget(home)
        nav.addStretch()
        nav.addWidget(export_btn)
        nav_widget = QtWidgets.QWidget()
        nav_widget.setLayout(nav)
        layout.addWidget(nav_widget)

    def _choose_export_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, t(self.main.lang, "label_outdir"))
        if path:
            self.export_dir.setText(path)

    def _export(self) -> None:
        npz_path = self.main.random_state.npz_path
        if npz_path is None:
            QtWidgets.QMessageBox.warning(
                self,
                t(self.main.lang, "msg_missing_title"),
                t(self.main.lang, "msg_missing_generate"),
            )
            return
        out_dir = Path(self.export_dir.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        formats = []
        if self.format_npz.isChecked():
            formats.append("npz")
        if self.format_json.isChecked():
            formats.append("json")
        if self.format_csv.isChecked():
            formats.append("csv")
        if self.format_h5.isChecked():
            formats.append("h5")
        if not formats:
            QtWidgets.QMessageBox.warning(
                self,
                t(self.main.lang, "msg_missing_title"),
                t(self.main.lang, "msg_select_format"),
            )
            return

        csv_index_value: Optional[int | str] = None
        if self.csv_index.text().strip():
            text = self.csv_index.text().strip()
            mid_token = t(self.main.lang, "value_mid")
            csv_index_value = "mid" if text in {"mid", mid_token} else int(text)
        if not self._confirm_export(npz_path, out_dir, formats):
            return
        self._start_export_task(npz_path, out_dir, formats, csv_index_value)

    def _confirm_export(self, npz_path: Path, out_dir: Path, formats: list[str]) -> bool:
        message = tf(
            self.main.lang,
            "msg_export_confirm",
            formats=", ".join(formats),
            outdir=out_dir,
            input_npz=npz_path.name,
        )
        result = QtWidgets.QMessageBox.question(
            self,
            t(self.main.lang, "msg_export_title"),
            message,
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return result == QtWidgets.QMessageBox.StandardButton.Ok

    def _start_export_task(
        self,
        npz_path: Path,
        out_dir: Path,
        formats: list[str],
        csv_index_value: Optional[int | str],
    ) -> None:
        self.set_enabled(False)
        self._show_export_dialog(t(self.main.lang, "msg_exporting"))
        worker: Optional[Worker] = None

        def task() -> Dict[str, Any]:
            if worker and worker.is_cancelled():
                return {"status": "cancelled"}
            paths: list[str] = []
            for fmt in formats:
                if worker and worker.is_cancelled():
                    return {"status": "cancelled"}
                out_path = out_dir / f"{npz_path.stem}.{fmt}"
                export_field(
                    npz_path,
                    fmt,
                    out_path,
                    {
                        "csv_slice_axis": self.csv_axis.currentText(),
                        "csv_slice_index": csv_index_value,
                    },
                )
                paths.append(str(out_path))
            return {"status": "done", "paths": paths}

        worker = Worker(task)
        self._export_worker = worker
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(self._on_export_error)
        self.main.thread_pool.start(worker)

    def _show_export_dialog(self, message: str) -> None:
        if self._export_dialog:
            self._finish_export_dialog()
        self._export_dialog = RenderProgressDialog(
            message,
            self._cancel_export,
            self.main.lang,
            t(self.main.lang, "dialog_export_title"),
            None,
        )
        self._export_dialog.show()

    def _finish_export_dialog(self) -> None:
        if self._export_dialog:
            self._export_dialog.allow_close()
            self._export_dialog.close()
            self._export_dialog = None
        self._export_worker = None
        self.set_enabled(True)

    def _cancel_export(self) -> None:
        if self._export_worker:
            self._export_worker.request_cancel()

    def _on_export_finished(self, payload: Dict[str, Any]) -> None:
        self._finish_export_dialog()
        status = payload.get("status")
        if status == "cancelled":
            QtWidgets.QMessageBox.information(
                self,
                t(self.main.lang, "msg_export_title"),
                t(self.main.lang, "msg_export_cancelled"),
            )
            return
        exported_paths = payload.get("paths", [])
        paths_text = "\n".join(str(p) for p in exported_paths)
        QtWidgets.QMessageBox.information(
            self,
            t(self.main.lang, "msg_export_title"),
            tf(self.main.lang, "msg_export_paths", paths=paths_text) if paths_text else t(self.main.lang, "msg_export_done"),
        )

    def _on_export_error(self, message: str) -> None:
        self._finish_export_dialog()
        QtWidgets.QMessageBox.critical(self, t(self.main.lang, "msg_error_title"), message)


class InvertConfigPage(PageBase):
    def __init__(self, main: "MainWindow", lang: str) -> None:
        super().__init__(main)
        self.lang = lang
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content.setMinimumWidth(760)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)
        content_layout.addWidget(self._build_inputs_group())
        content_layout.addWidget(self._build_colorbar_group())
        content_layout.addWidget(self._build_mask_group())
        content_layout.addWidget(self._build_output_group())
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        nav = self._build_nav(
            home_cb=lambda: self.main.show_page("home"),
            next_cb=self._go_next,
            show_back=False,
            show_run=False,
            show_next=True,
        )
        layout.addWidget(nav)

    def _build_inputs_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_inputs", attr="setTitle")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.field_image = QtWidgets.QLineEdit()
        self.cbar_image = QtWidgets.QLineEdit()
        self._register_tooltip(self.field_image, "tip_field_image")
        self._register_tooltip(self.cbar_image, "tip_cbar_image")
        field_btn = QtWidgets.QPushButton()
        cbar_btn = QtWidgets.QPushButton()
        self._register_text(field_btn, "btn_browse")
        self._register_text(cbar_btn, "btn_browse")
        field_btn.clicked.connect(lambda: self._choose_file(self.field_image))
        cbar_btn.clicked.connect(lambda: self._choose_file(self.cbar_image))

        field_layout = QtWidgets.QHBoxLayout()
        field_layout.setSpacing(8)
        field_layout.addWidget(self.field_image)
        field_layout.addWidget(field_btn)
        field_widget = QtWidgets.QWidget()
        field_widget.setLayout(field_layout)

        cbar_layout = QtWidgets.QHBoxLayout()
        cbar_layout.setSpacing(8)
        cbar_layout.addWidget(self.cbar_image)
        cbar_layout.addWidget(cbar_btn)
        cbar_widget = QtWidgets.QWidget()
        cbar_widget.setLayout(cbar_layout)

        form.addRow(self._make_label("label_field_image"), field_widget)
        form.addRow(self._make_label("label_cbar_image"), cbar_widget)
        return group

    def _build_colorbar_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_colorbar", attr="setTitle")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        self.cbar_min = QtWidgets.QDoubleSpinBox()
        self.cbar_max = QtWidgets.QDoubleSpinBox()
        self.cbar_min.setRange(-360.0, 360.0)
        self.cbar_max.setRange(-360.0, 360.0)
        self.cbar_min.setValue(-45.0)
        self.cbar_max.setValue(45.0)
        self._register_tooltip(self.cbar_min, "tip_cbar_min")
        self._register_tooltip(self.cbar_max, "tip_cbar_max")

        self.cbar_direction = QtWidgets.QComboBox()
        self._register_combo_items(
            self.cbar_direction,
            [
                ("top_high", "combo_cbar_top_high"),
                ("top_low", "combo_cbar_top_low"),
            ],
        )
        self._register_tooltip(self.cbar_direction, "tip_cbar_direction")

        form.addRow(self._make_label("label_cbar_min"), self.cbar_min)
        form.addRow(self._make_label("label_cbar_max"), self.cbar_max)
        form.addRow(self._make_label("label_cbar_direction"), self.cbar_direction)
        return group

    def _build_mask_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_mask", attr="setTitle")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        self.defect_policy = QtWidgets.QComboBox()
        self._register_combo_items(
            self.defect_policy,
            [
                ("nan", "combo_defect_policy_nan"),
                ("zero", "combo_defect_policy_zero"),
                ("fill", "combo_defect_policy_fill"),
            ],
        )
        self.color_tolerance = QtWidgets.QDoubleSpinBox()
        self.color_tolerance.setRange(0.0, 255.0)
        self.color_tolerance.setValue(50.0)
        self.defect_threshold = QtWidgets.QSpinBox()
        self.defect_threshold.setRange(0, 255)
        self.defect_threshold.setValue(8)
        self._register_tooltip(self.defect_policy, "tip_defect_policy")
        self._register_tooltip(self.color_tolerance, "tip_color_tolerance")
        self._register_tooltip(self.defect_threshold, "tip_defect_threshold")
        form.addRow(self._make_label("label_defect_policy"), self.defect_policy)
        form.addRow(self._make_label("label_color_tolerance"), self.color_tolerance)
        form.addRow(self._make_label("label_defect_threshold"), self.defect_threshold)
        return group

    def _build_output_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_output", attr="setTitle")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        self.outdir = QtWidgets.QLineEdit(str(OUTPUT_DIR))
        self._prefix_default = t(self.main.lang, "default_prefix_invert")
        self.prefix = QtWidgets.QLineEdit(self._prefix_default)
        self._register_tooltip(self.outdir, "tip_outdir")
        self._register_tooltip(self.prefix, "tip_prefix")

        def update_prefix_default(lang: str) -> None:
            new_default = t(lang, "default_prefix_invert")
            if self.prefix.text() == self._prefix_default:
                self.prefix.setText(new_default)
            self._prefix_default = new_default

        self._translations.append(update_prefix_default)
        browse_btn = QtWidgets.QPushButton()
        self._register_text(browse_btn, "btn_browse")
        browse_btn.clicked.connect(self._choose_outdir)
        out_layout = QtWidgets.QHBoxLayout()
        out_layout.setSpacing(8)
        out_layout.addWidget(self.outdir)
        out_layout.addWidget(browse_btn)
        out_widget = QtWidgets.QWidget()
        out_widget.setLayout(out_layout)
        form.addRow(self._make_label("label_outdir"), out_widget)
        form.addRow(self._make_label("label_prefix"), self.prefix)
        return group

    def _choose_outdir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, t(self.main.lang, "label_outdir"))
        if path:
            self.outdir.setText(path)

    def _choose_file(self, target: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, t(self.main.lang, "btn_browse"))
        if path:
            target.setText(path)

    def _go_next(self) -> None:
        if not self.field_image.text().strip() or not self.cbar_image.text().strip():
            QtWidgets.QMessageBox.warning(
                self,
                t(self.main.lang, "msg_missing_title"),
                t(self.main.lang, "msg_missing_inputs"),
            )
            return
        if self.cbar_min.value() >= self.cbar_max.value():
            QtWidgets.QMessageBox.warning(
                self,
                t(self.main.lang, "msg_validation_title"),
                t(self.main.lang, "msg_cbar_min_validation"),
            )
            return
        self.main.invert_state.config = self._build_config()
        self.main.invert_state.outdir = Path(self.outdir.text())
        self.main.invert_state.prefix = self.prefix.text().strip() or t(self.main.lang, "default_prefix_invert")
        self.main.show_page("invert_visualize")

    def _build_config(self) -> FromImagesConfig:
        direction = self.cbar_direction.currentData() or self.cbar_direction.currentText()
        direction = "top_high" if direction in {"top=high", "top_high"} else "top_low"
        return FromImagesConfig(
            field_image=Path(self.field_image.text()),
            cbar_image=Path(self.cbar_image.text()),
            cbar_min=self.cbar_min.value(),
            cbar_max=self.cbar_max.value(),
            cbar_direction=direction,
            defect_threshold=self.defect_threshold.value(),
            color_tolerance=self.color_tolerance.value(),
            defect_policy=self.defect_policy.currentData() or self.defect_policy.currentText(),
            orientation_format="matrix",
            mask_min_area=9,
            mask_morph="open",
            mask_kernel=3,
        )


class InvertVisualizePage(PageBase):
    def __init__(self, main: "MainWindow", lang: str) -> None:
        super().__init__(main)
        self.lang = lang
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.status_label = QtWidgets.QLabel()
        self._status_key = "status_ready"
        self._status_args: dict[str, Any] = {}
        self._translations.append(
            lambda current_lang: self.status_label.setText(
                tf(current_lang, self._status_key, **self._status_args)
            )
        )
        self.apply_language(self.main.lang)
        layout.addWidget(self.status_label)

        viewer_layout = QtWidgets.QHBoxLayout()
        viewer_layout.setSpacing(8)
        viewer_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        preview_side = 320
        self.field_view = QtWidgets.QLabel()
        self.mask_view = QtWidgets.QLabel()
        for view in (self.field_view, self.mask_view):
            view.setFixedSize(preview_side, preview_side)
            view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            view.setScaledContents(False)
            view.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        viewer_layout.addWidget(self.field_view)
        viewer_layout.addWidget(self.mask_view)
        viewer_container = QtWidgets.QWidget()
        viewer_container.setLayout(viewer_layout)
        viewer_container.setContentsMargins(0, 0, 0, 0)
        viewer_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(viewer_container)

        self.meta_label = QtWidgets.QLabel("-")
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(6)
        back = QtWidgets.QPushButton()
        self._register_text(back, "btn_back")
        back.clicked.connect(lambda: self.main.show_page("invert_config"))
        home = QtWidgets.QPushButton()
        self._register_text(home, "btn_home")
        home.clicked.connect(lambda: self.main.show_page("home"))
        run = QtWidgets.QPushButton()
        self._register_text(run, "btn_run")
        run.clicked.connect(self._run_inversion)
        next_btn = QtWidgets.QPushButton()
        self._register_text(next_btn, "btn_next")
        next_btn.clicked.connect(lambda: self.main.show_page("invert_export"))
        nav.addWidget(back)
        nav.addWidget(home)
        nav.addStretch()
        nav.addWidget(run)
        nav.addWidget(next_btn)
        nav_widget = QtWidgets.QWidget()
        nav_widget.setLayout(nav)
        layout.addWidget(nav_widget)

    def _run_inversion(self) -> None:
        config = self.main.invert_state.config
        if config is None:
            QtWidgets.QMessageBox.warning(
                self,
                t(self.main.lang, "msg_missing_title"),
                t(self.main.lang, "msg_missing_inputs_config"),
            )
            return
        self._set_status("status_running")
        self.set_enabled(False)

        def task() -> Dict[str, Any]:
            field = invert_field_from_images(config)
            meta = {
                "schema_version": "1.0",
                "source": "from-images",
                "dim": 2,
                "shape": list(field["theta_deg"].shape),
                "mask_ratio": float(np.mean(field["mask"].astype(bool))),
                "cbar_min": config.cbar_min,
                "cbar_max": config.cbar_max,
            }
            field["meta"] = meta
            outdir = self.main.invert_state.outdir or OUTPUT_DIR
            outdir.mkdir(parents=True, exist_ok=True)
            prefix = self.main.invert_state.prefix
            output_path = outdir / f"{prefix}.npz"
            output_path = write_npz(field, output_path)
            preview_dir = OUTPUT_DIR / "preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            vmin, vmax = resolve_global_vmin_vmax(field)
            theta_2d, mask_2d = compute_slice_arrays_from_npz(output_path, "z", 0)
            field_path = preview_dir / f"{output_path.stem}_field.png"
            mask_path = preview_dir / f"{output_path.stem}_mask.png"
            render_field_and_mask_png(
                theta_2d,
                mask_2d,
                field_path,
                mask_path,
                vmin,
                vmax,
                PREVIEW_STYLE_CFG,
            )
            return {
                "npz": output_path,
                "preview_dir": preview_dir,
                "field_path": field_path,
                "mask_path": mask_path,
                "meta": meta,
            }

        worker = Worker(task)
        worker.signals.finished.connect(self._on_inverted)
        worker.signals.error.connect(self._on_error)
        self.main.thread_pool.start(worker)

    def _on_inverted(self, payload: Dict[str, Any]) -> None:
        self.main.invert_state.npz_path = payload["npz"]
        self.main.invert_state.preview_dir = payload["preview_dir"]
        self.main.invert_state.meta = payload["meta"]
        self._load_preview(payload["field_path"], payload["mask_path"])
        meta = payload["meta"]
        self._meta_cache = meta
        self._refresh_meta_label(self.main.lang)
        self._set_status("status_done")
        self.set_enabled(True)

    def _on_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, t(self.main.lang, "msg_error_title"), message)
        self._set_status("status_error")
        self.set_enabled(True)

    def _set_status(self, key: str, **kwargs: Any) -> None:
        self._status_key = key
        self._status_args = kwargs
        self.status_label.setText(tf(self.main.lang, key, **kwargs))

    def _refresh_meta_label(self, lang: str) -> None:
        if not self._meta_cache:
            return
        self.meta_label.setText(
            tf(
                lang,
                "msg_meta_invert",
                dim=self._meta_cache.get("dim"),
                shape=self._meta_cache.get("shape"),
                mask_ratio=self._meta_cache.get("mask_ratio"),
            )
        )

    def _load_preview(self, field_path: Path, mask_path: Path) -> None:
        pixmap = QtGui.QPixmap(str(field_path))
        field_scaled = pixmap.scaled(
            self.field_view.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.field_view.setPixmap(field_scaled)
        mask_pixmap = QtGui.QPixmap(str(mask_path))
        mask_scaled = mask_pixmap.scaled(
            self.mask_view.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.mask_view.setPixmap(mask_scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.main.invert_state.preview_dir and self.main.invert_state.npz_path:
            preview_dir = self.main.invert_state.preview_dir
            prefix = self.main.invert_state.npz_path.stem
            field_path = preview_dir / f"{prefix}_field.png"
            mask_path = preview_dir / f"{prefix}_mask.png"
            if field_path.exists() and mask_path.exists():
                self._load_preview(field_path, mask_path)


class InvertExportPage(PageBase):
    def __init__(self, main: "MainWindow", lang: str) -> None:
        super().__init__(main)
        self.lang = lang
        self._export_worker: Optional[Worker] = None
        self._export_dialog: Optional[RenderProgressDialog] = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        group = QtWidgets.QGroupBox()
        self._register_text(group, "group_export", attr="setTitle")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

        self.export_dir = QtWidgets.QLineEdit(str(OUTPUT_DIR / "exports"))
        self._register_tooltip(self.export_dir, "tip_export_dir")
        browse_btn = QtWidgets.QPushButton()
        self._register_text(browse_btn, "btn_browse")
        browse_btn.clicked.connect(self._choose_export_dir)
        out_layout = QtWidgets.QHBoxLayout()
        out_layout.setSpacing(6)
        out_layout.addWidget(self.export_dir)
        out_layout.addWidget(browse_btn)
        out_widget = QtWidgets.QWidget()
        out_widget.setLayout(out_layout)

        self.format_npz = QtWidgets.QCheckBox("NPZ")
        self.format_json = QtWidgets.QCheckBox("JSON")
        self.format_csv = QtWidgets.QCheckBox("CSV")
        self.format_h5 = QtWidgets.QCheckBox("HDF5")
        self._register_tooltip(self.format_npz, "tip_export_npz")
        self._register_tooltip(self.format_json, "tip_export_json")
        self._register_tooltip(self.format_csv, "tip_export_csv")
        self._register_tooltip(self.format_h5, "tip_export_h5")
        self.format_npz.setChecked(True)
        format_layout = QtWidgets.QHBoxLayout()
        format_layout.setSpacing(6)
        for widget in (self.format_npz, self.format_json, self.format_csv, self.format_h5):
            format_layout.addWidget(widget)
        format_widget = QtWidgets.QWidget()
        format_widget.setLayout(format_layout)

        form.addRow(self._make_label("label_outdir"), out_widget)
        form.addRow(self._make_label("label_formats"), format_widget)

        layout.addWidget(group)
        layout.addStretch()

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(6)
        back = QtWidgets.QPushButton()
        self._register_text(back, "btn_back")
        back.clicked.connect(lambda: self.main.show_page("invert_visualize"))
        home = QtWidgets.QPushButton()
        self._register_text(home, "btn_home")
        home.clicked.connect(lambda: self.main.show_page("home"))
        export_btn = QtWidgets.QPushButton()
        self._register_text(export_btn, "btn_export")
        export_btn.clicked.connect(self._export)
        nav.addWidget(back)
        nav.addWidget(home)
        nav.addStretch()
        nav.addWidget(export_btn)
        nav_widget = QtWidgets.QWidget()
        nav_widget.setLayout(nav)
        layout.addWidget(nav_widget)

    def _choose_export_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, t(self.main.lang, "label_outdir"))
        if path:
            self.export_dir.setText(path)

    def _export(self) -> None:
        npz_path = self.main.invert_state.npz_path
        if npz_path is None:
            QtWidgets.QMessageBox.warning(
                self,
                t(self.main.lang, "msg_missing_title"),
                t(self.main.lang, "msg_missing_invert"),
            )
            return
        out_dir = Path(self.export_dir.text())
        out_dir.mkdir(parents=True, exist_ok=True)
        formats = []
        if self.format_npz.isChecked():
            formats.append("npz")
        if self.format_json.isChecked():
            formats.append("json")
        if self.format_csv.isChecked():
            formats.append("csv")
        if self.format_h5.isChecked():
            formats.append("h5")
        if not formats:
            QtWidgets.QMessageBox.warning(
                self,
                t(self.main.lang, "msg_missing_title"),
                t(self.main.lang, "msg_select_format"),
            )
            return
        if not self._confirm_export(npz_path, out_dir, formats):
            return
        self._start_export_task(npz_path, out_dir, formats)

    def _confirm_export(self, npz_path: Path, out_dir: Path, formats: list[str]) -> bool:
        message = tf(
            self.main.lang,
            "msg_export_confirm",
            formats=", ".join(formats),
            outdir=out_dir,
            input_npz=npz_path.name,
        )
        result = QtWidgets.QMessageBox.question(
            self,
            t(self.main.lang, "msg_export_title"),
            message,
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return result == QtWidgets.QMessageBox.StandardButton.Ok

    def _start_export_task(self, npz_path: Path, out_dir: Path, formats: list[str]) -> None:
        self.set_enabled(False)
        self._show_export_dialog(t(self.main.lang, "msg_exporting"))
        worker: Optional[Worker] = None

        def task() -> Dict[str, Any]:
            if worker and worker.is_cancelled():
                return {"status": "cancelled"}
            paths: list[str] = []
            for fmt in formats:
                if worker and worker.is_cancelled():
                    return {"status": "cancelled"}
                out_path = out_dir / f"{npz_path.stem}.{fmt}"
                export_field(npz_path, fmt, out_path, {})
                paths.append(str(out_path))
            return {"status": "done", "paths": paths}

        worker = Worker(task)
        self._export_worker = worker
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.error.connect(self._on_export_error)
        self.main.thread_pool.start(worker)

    def _show_export_dialog(self, message: str) -> None:
        if self._export_dialog:
            self._finish_export_dialog()
        self._export_dialog = RenderProgressDialog(
            message,
            self._cancel_export,
            self.main.lang,
            t(self.main.lang, "dialog_export_title"),
            None,
        )
        self._export_dialog.show()

    def _finish_export_dialog(self) -> None:
        if self._export_dialog:
            self._export_dialog.allow_close()
            self._export_dialog.close()
            self._export_dialog = None
        self._export_worker = None
        self.set_enabled(True)

    def _cancel_export(self) -> None:
        if self._export_worker:
            self._export_worker.request_cancel()

    def _on_export_finished(self, payload: Dict[str, Any]) -> None:
        self._finish_export_dialog()
        status = payload.get("status")
        if status == "cancelled":
            QtWidgets.QMessageBox.information(
                self,
                t(self.main.lang, "msg_export_title"),
                t(self.main.lang, "msg_export_cancelled"),
            )
            return
        exported_paths = payload.get("paths", [])
        paths_text = "\n".join(str(p) for p in exported_paths)
        QtWidgets.QMessageBox.information(
            self,
            t(self.main.lang, "msg_export_title"),
            tf(self.main.lang, "msg_export_paths", paths=paths_text) if paths_text else t(self.main.lang, "msg_export_done"),
        )

    def _on_export_error(self, message: str) -> None:
        self._finish_export_dialog()
        QtWidgets.QMessageBox.critical(self, t(self.main.lang, "msg_error_title"), message)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, lang: str = "zh") -> None:
        super().__init__()
        self.lang = lang
        self.random_state = RandomRunState()
        self.invert_state = InvertRunState()
        self.thread_pool = QtCore.QThreadPool.globalInstance()

        self.setWindowTitle(t(lang, "app_title"))
        self.resize(980, 620)
        self.setMinimumSize(860, 560)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.stack = QtWidgets.QStackedWidget()

        header = QtWidgets.QHBoxLayout()
        header.addStretch()
        self.language_label = QtWidgets.QLabel()
        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItem(t(lang, "label_language_zh"), "zh")
        self.language_combo.addItem(t(lang, "label_language_en"), "en")
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        header.addWidget(self.language_label)
        header.addWidget(self.language_combo)
        header_widget = QtWidgets.QWidget()
        header_widget.setLayout(header)
        layout.addWidget(header_widget)

        self.pages: Dict[str, QtWidgets.QWidget] = {}
        self._add_page("home", HomePage(self, lang))
        self._add_page("random_config", RandomConfigPage(self, lang))
        self._add_page("random_visualize", RandomVisualizePage(self, lang))
        self._add_page("random_export", RandomExportPage(self, lang))
        self._add_page("invert_config", InvertConfigPage(self, lang))
        self._add_page("invert_visualize", InvertVisualizePage(self, lang))
        self._add_page("invert_export", InvertExportPage(self, lang))

        layout.addWidget(self.stack)
        layout.addWidget(self._build_log_panel())
        self.setCentralWidget(central)

        self.language_combo.setCurrentIndex(0 if lang == "zh" else 1)
        self.set_language(lang)
        self.show_page("home")

    def _add_page(self, name: str, widget: QtWidgets.QWidget) -> None:
        self.pages[name] = widget
        self.stack.addWidget(widget)

    def show_page(self, name: str) -> None:
        page = self.pages[name]
        self.stack.setCurrentWidget(page)
        sync = getattr(page, "sync_from_state", None)
        if callable(sync):
            sync()

    def _build_log_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QGroupBox()
        panel.setTitle(t(self.lang, "panel_logs"))
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        emitter = LogEmitter()
        handler = QtLogHandler(emitter)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        file_handler_exists = any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOG_FILE) for h in root_logger.handlers)
        if not file_handler_exists:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)
        emitter.message.connect(self.log_view.appendPlainText)
        self.log_panel = panel
        return panel

    def _on_language_changed(self) -> None:
        lang = self.language_combo.currentData()
        if isinstance(lang, str):
            self.set_language(lang)

    def set_language(self, lang: str) -> None:
        if lang == self.lang:
            self.language_label.setText(t(lang, "label_language"))
            return
        self.lang = lang
        self.setWindowTitle(t(lang, "app_title"))
        self.language_label.setText(t(lang, "label_language"))
        self.language_combo.setItemText(0, t(lang, "label_language_zh"))
        self.language_combo.setItemText(1, t(lang, "label_language_en"))
        for page in self.pages.values():
            if isinstance(page, PageBase):
                page.apply_language(lang)
        if hasattr(self, "log_panel"):
            self.log_panel.setTitle(t(lang, "panel_logs"))


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()