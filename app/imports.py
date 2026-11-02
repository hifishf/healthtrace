"""健康数据导入：CSV 解析 + 校验（指标代码/单位/日期）+ 覆盖入库。

第 9 周冻结验收：非法 CSV（错单位/错日期）被校验拦截；
合法行按 metric_records 的 (user_id, metric_code, recorded_date) 覆盖语义入库。
数据访问走 crud（session 内部管理），本模块不直接写 SQL。
"""
import csv
import math
from datetime import date
from uuid import UUID

from app import crud

# CSV 必需列（冻结）
REQUIRED_COLUMNS = ("metric_code", "value", "unit", "date")

# 纯格式校验错误码 -> 中文提示
_ERROR_MESSAGES = {
    "EMPTY_METRIC_CODE": "指标代码为空",
    "EMPTY_UNIT": "单位为空",
    "INVALID_DATE": "日期不合法（应为 YYYY-MM-DD）",
    "INVALID_VALUE": "数值不合法",
}


def parse_metric_csv(text: str) -> tuple[list[dict], list[dict]]:
    """解析 CSV 文本 -> (rows, file_errors)。

    表头必须包含 metric_code/value/unit/date 四列，缺列返回文件级错误；
    数据行按列名取值，缺失单元格留空，交由后续行校验拦截。
    """
    if not text.strip():
        return [], [{"error_code": "EMPTY_CSV", "message": "CSV 内容为空"}]

    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        return [], [{"error_code": "EMPTY_CSV", "message": "CSV 内容为空"}]

    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        return [], [{"error_code": "MISSING_COLUMN", "message": f"缺少列：{', '.join(missing)}"}]

    rows = [{column: row.get(column) for column in REQUIRED_COLUMNS} for row in reader]
    return rows, []


def validate_metric_row(raw: dict) -> tuple[dict | None, str | None]:
    """纯格式校验单行：指标代码/单位非空、日期合法（YYYY-MM-DD）、数值有限。

    返回 (normalized, None) 或 (None, error_code)；
    normalized 含 metric_code/value/unit/recorded_date 四个规范化字段。
    """
    metric_code = str(raw.get("metric_code") or "").strip()
    if not metric_code:
        return None, "EMPTY_METRIC_CODE"

    unit = str(raw.get("unit") or "").strip()
    if not unit:
        return None, "EMPTY_UNIT"

    date_text = str(raw.get("date") or raw.get("recorded_date") or "").strip()
    try:
        recorded_date = date.fromisoformat(date_text)
    except ValueError:
        return None, "INVALID_DATE"

    try:
        value = float(raw.get("value"))
    except (TypeError, ValueError):
        return None, "INVALID_VALUE"
    if not math.isfinite(value):
        return None, "INVALID_VALUE"

    return {
        "metric_code": metric_code,
        "value": value,
        "unit": unit,
        "recorded_date": recorded_date,
    }, None


def _message_for(error_code: str) -> str:
    """错误码 -> 中文提示。"""
    return _ERROR_MESSAGES.get(error_code, error_code)


async def import_metric_rows(
    user_id: str | UUID,
    source: str,
    rows: list[dict],
) -> dict:
    """校验并批量入库指标行，返回 {"ok": True, "imported": int, "rejected": [...]}。

    校验两层：纯格式（validate_metric_row）+ 单位与参考范围比对（错单位拦截）。
    合法行按 (user_id, metric_code, recorded_date) 覆盖写入 metric_records；
    被拒行带 0 基 index 与 error_code，不阻断其余合法行入库。
    """
    uid = UUID(user_id) if isinstance(user_id, str) else user_id

    rejected: list[dict] = []
    valid: list[tuple[int, dict]] = []
    for index, raw in enumerate(rows):
        normalized, error_code = validate_metric_row(raw)
        if error_code:
            rejected.append(
                {"index": index, "error_code": error_code, "message": _message_for(error_code)}
            )
        else:
            valid.append((index, normalized))

    accepted: list[dict] = []
    if valid:
        codes = sorted({norm["metric_code"] for _, norm in valid})
        units_by_code = await crud.get_reference_units(codes)
        for index, norm in valid:
            allowed = units_by_code.get(norm["metric_code"])
            if allowed and norm["unit"] not in allowed:
                rejected.append(
                    {
                        "index": index,
                        "error_code": "UNIT_MISMATCH",
                        "message": (
                            f"指标 {norm['metric_code']} 的单位 {norm['unit']} "
                            "不在参考范围允许的单位内"
                        ),
                    }
                )
            else:
                accepted.append(norm)

    if accepted:
        await crud.bulk_upsert_metrics(uid, accepted, source)

    return {"ok": True, "imported": len(accepted), "rejected": rejected}
