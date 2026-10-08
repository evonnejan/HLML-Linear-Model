"""
HL01 推論套件：input 檢查與 Predictor 預測入口。

匯入這個套件不會載入 torch；第一次使用 Predictor 時才載入。
"""

from .validation import InputValidationError, ValidatedWindow, validate_input

__all__ = ["InputValidationError", "ValidatedWindow", "validate_input", "Predictor"]


def __getattr__(name):
    if name == "Predictor":
        from .predictor import Predictor
        return Predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
