"""Streamlit 界面：指标表单（第 1 周）+ 对话式记录（第 2 周，SSE 流式）。"""
import datetime
import json
import os

import requests
import streamlit as st

# FastAPI 网关地址（默认本机 8000，可用环境变量覆盖）
API_BASE_URL = os.getenv("HEALTHTRACE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="HealthTrace", layout="centered")
st.title("HealthTrace")
st.caption("指标表单（第 1 周）+ 对话式记录（第 2 周，SSE 流式）")

# ===== 第 1 周：指标表单 =====
with st.form("metric_form", clear_on_submit=False):
    metric_code = st.text_input("指标代码", value="sleep_duration")
    value = st.number_input("数值", value=6.0, step=0.5, min_value=0.0)
    unit = st.text_input("单位", value="h")
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


# ===== 第 2 周：对话式记录（SSE 流式）=====
st.divider()
st.subheader("对话式记录")
st.caption("试着说：昨晚睡了 6 小时")


def chat_stream(text: str):
    """消费 POST /chat/stream，逐 token 产出回复文本（供 st.write_stream）。"""
    with requests.post(
        f"{API_BASE_URL}/chat/stream",
        json={"text": text},
        stream=True,
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            try:
                event = json.loads(raw[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if event.get("event") == "chat":
                yield event.get("data", "")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    st.chat_message(role).write(content)

if user_input := st.chat_input("跟我说点什么"):
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(chat_stream(user_input))
        except requests.RequestException as exc:
            reply = f"请求失败（请确认 FastAPI 已在 8000 端口启动）：{exc}"
            st.error(reply)
    st.session_state.chat_history.append(("assistant", reply))
