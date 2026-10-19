"""Streamlit 记录界面：把表单提交到 FastAPI 的 POST /metrics。"""
import datetime
import os

import requests
import streamlit as st

# FastAPI 网关地址（默认本机 8000，可用环境变量覆盖）
API_BASE_URL = os.getenv("HEALTHTRACE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="HealthTrace 记录", layout="centered")
st.title("HealthTrace · 记录一条健康数据")
st.caption("数据写入 PostgreSQL metric_records；同日期同指标为覆盖语义。")

with st.form("metric_form", clear_on_submit=False):
    metric_code = st.text_input("指标代码", value="sleep_duration", help="例如 sleep_duration")
    value = st.number_input("数值", value=6.0, step=0.5, min_value=0.0)
    unit = st.text_input("单位", value="h", help="例如 h / kg / 次")
    recorded_date = st.date_input("日期", value=datetime.date.today())
    submitted = st.form_submit_button("提交")

if submitted:
    payload = {
        "metric_code": metric_code,
        "value": value,
        "unit": unit,
        "recorded_date": recorded_date.isoformat(),
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/metrics", json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        st.success(f"已记录：{metric_code} = {value} {unit}（id={data['id']}）")
    except requests.RequestException as exc:
        st.error(f"提交失败（请确认 FastAPI 已在 8000 端口启动）：{exc}")
