"""报告导入（imports）的确定性断言：CSV 解析 / 校验拦截 / 覆盖入库。"""
from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import delete, insert

from app import crud
from app.db import async_session
from app.imports import import_metric_rows, parse_metric_csv, validate_metric_row
from app.models import LabReferenceRange, MetricRecord

# 测试专用 user_id + metric_code，测完清理，不污染默认用户与种子数据
TEST_USER_ID = UUID("00000000-0000-0000-0000-00000000fffe")
TEST_METRIC = "test_ldl_cholesterol"


async def _seed_reference_range(unit: str = "mmol/L") -> None:
    """写入测试指标的参考范围（先清后插，保证幂等）。"""
    async with async_session() as session:
        await session.execute(
            delete(LabReferenceRange).where(LabReferenceRange.metric_code == TEST_METRIC)
        )
        await session.execute(
            insert(LabReferenceRange).values(
                metric_code=TEST_METRIC,
                unit=unit,
                low=None,
                high=Decimal("3.4"),
            )
        )
        await session.commit()


async def _clear_metrics() -> None:
    """清理测试用户的历史指标记录（保留参考范围）。"""
    async with async_session() as session:
        await session.execute(
            delete(MetricRecord).where(MetricRecord.user_id == TEST_USER_ID)
        )
        await session.commit()


async def _teardown() -> None:
    """收尾清理：指标记录 + 参考范围一起清掉，恢复初始状态。"""
    async with async_session() as session:
        await session.execute(
            delete(MetricRecord).where(MetricRecord.user_id == TEST_USER_ID)
        )
        await session.execute(
            delete(LabReferenceRange).where(LabReferenceRange.metric_code == TEST_METRIC)
        )
        await session.commit()


async def test_import_valid_rows_and_overwrite() -> None:
    """合法三行全部入库，重复导入同日同指标覆盖为单行新值。"""
    await _seed_reference_range("mmol/L")
    await _clear_metrics()
    try:
        rows = [
            {"metric_code": TEST_METRIC, "value": "3.1", "unit": "mmol/L", "date": "2024-01-15"},
            {"metric_code": TEST_METRIC, "value": "3.4", "unit": "mmol/L", "date": "2025-01-15"},
            {"metric_code": TEST_METRIC, "value": "3.8", "unit": "mmol/L", "date": "2026-01-15"},
        ]
        result = await import_metric_rows(str(TEST_USER_ID), "csv", rows)
        assert result["ok"] is True
        assert result["imported"] == 3
        assert result["rejected"] == []

        # 覆盖语义：同日同指标重复导入 -> 仍 3 行、2026-01-15 值被覆盖为 4.0
        again = await import_metric_rows(
            str(TEST_USER_ID),
            "csv",
            [{"metric_code": TEST_METRIC, "value": "4.0", "unit": "mmol/L", "date": "2026-01-15"}],
        )
        assert again["imported"] == 1

        metrics = await crud.query_metrics(TEST_USER_ID, metric_code=TEST_METRIC)
        assert len(metrics) == 3
        by_date = {m["recorded_date"]: m["value"] for m in metrics}
        assert by_date["2026-01-15"] == 4.0
    finally:
        await _teardown()


async def test_import_rejects_wrong_unit_and_date() -> None:
    """断言：错单位与错日期被校验拦截，合法行不因坏行整体失败。"""
    await _seed_reference_range("mmol/L")
    await _clear_metrics()
    try:
        result = await import_metric_rows(
            str(TEST_USER_ID),
            "csv",
            [
                {"metric_code": TEST_METRIC, "value": "3.1", "unit": "mg/dL", "date": "2024-01-15"},
                {"metric_code": TEST_METRIC, "value": "3.4", "unit": "mmol/L", "date": "2026-02-30"},
                {"metric_code": TEST_METRIC, "value": "3.6", "unit": "mmol/L", "date": "2026-01-15"},
            ],
        )
        assert result["imported"] == 1
        codes = {r["error_code"] for r in result["rejected"]}
        assert codes == {"UNIT_MISMATCH", "INVALID_DATE"}
    finally:
        await _teardown()


def test_parse_metric_csv_valid_and_missing_column() -> None:
    """CSV 解析：合法表头逐行取值；缺列返回 MISSING_COLUMN 文件级错误。"""
    rows, errors = parse_metric_csv(
        "metric_code,value,unit,date\nldl,3.1,mmol/L,2024-01-15\n"
    )
    assert errors == []
    assert rows == [
        {"metric_code": "ldl", "value": "3.1", "unit": "mmol/L", "date": "2024-01-15"}
    ]

    rows2, errors2 = parse_metric_csv("metric_code,value,unit\nldl,3.1,mmol/L\n")
    assert rows2 == []
    assert errors2[0]["error_code"] == "MISSING_COLUMN"


def test_validate_metric_row_field_errors() -> None:
    """纯格式校验：空指标码/空单位/非法数值/非法日期各返回对应错误码。"""
    assert validate_metric_row(
        {"metric_code": "", "value": "3", "unit": "mmol/L", "date": "2024-01-15"}
    )[1] == "EMPTY_METRIC_CODE"
    assert validate_metric_row(
        {"metric_code": "ldl", "value": "x", "unit": "mmol/L", "date": "2024-01-15"}
    )[1] == "INVALID_VALUE"
    assert validate_metric_row(
        {"metric_code": "ldl", "value": "3", "unit": "", "date": "2024-01-15"}
    )[1] == "EMPTY_UNIT"
    assert validate_metric_row(
        {"metric_code": "ldl", "value": "3", "unit": "mmol/L", "date": "2024/01/15"}
    )[1] == "INVALID_DATE"

    normalized, error = validate_metric_row(
        {"metric_code": " ldl ", "value": "3.1", "unit": " mmol/L ", "date": "2024-01-15"}
    )
    assert error is None
    assert normalized["metric_code"] == "ldl"
    assert normalized["unit"] == "mmol/L"
    assert normalized["value"] == 3.1
    assert normalized["recorded_date"] == date(2024, 1, 15)
