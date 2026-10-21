"""Streamlit 界面：指标表单 + 对话式记录 + 追问分诊（第 1~3 周）。"""
import datetime
import json
import os

import requests
import streamlit as st

# FastAPI 网关地址（默认本机 8000，可用环境变量覆盖）
API_BASE_URL = os.getenv("HEALTHTRACE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="HealthTrace", layout="centered")
st.title("HealthTrace")
st.caption("指标表单（第 1 周）+ 对话式记录（第 2 周）+ 追问分诊（第 3 周）")

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


# ===== 第 3 周：追问分诊（多轮）=====
st.divider()
st.subheader("追问分诊")
st.caption("描述症状，系统最多追问 3 轮后给出分级建议；试试：最近总是累 / 胸口疼得喘不上气 / 我爸爸胸痛")

if "triage_thread_id" not in st.session_state:
    st.session_state.triage_thread_id = None
if "triage_messages" not in st.session_state:
    st.session_state.triage_messages = []
if "triage_metrics" not in st.session_state:
    st.session_state.triage_metrics = None

for role, content in st.session_state.triage_messages:
    st.chat_message(role).write(content)

# 开发者指标（不进病人对话，单独折叠显示，用于演示候选集逐轮收敛）
with st.expander("开发者指标：候选集收敛", expanded=False):
    metrics = st.session_state.triage_metrics
    if not metrics:
        st.caption("暂无。发送一次症状描述后，这里显示候选集逐轮收敛情况。")
    else:
        col1, col2 = st.columns(2)
        col1.metric("候选疾病总数", metrics["candidate_count"])
        col2.metric("高置信档", metrics["top_tier_count"])
        if metrics["top"]:
            st.write("命中数最高的候选：")
            for name, hits in metrics["top"]:
                st.write(f"- {name}（命中 {hits} 个症状）")

with st.form("triage_form", clear_on_submit=True):
    triage_text = st.text_input("症状描述 / 回答追问")
    triage_sent = st.form_submit_button("发送")

if triage_sent and triage_text.strip():
    user_text = triage_text.strip()
    st.session_state.triage_messages.append(("user", user_text))
    payload = {"text": user_text}
    if st.session_state.triage_thread_id:
        payload["thread_id"] = st.session_state.triage_thread_id
    try:
        resp = requests.post(f"{API_BASE_URL}/triage", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidate_conditions") or []
        st.session_state.triage_metrics = {
            "candidate_count": data.get("candidate_count", 0),
            "top_tier_count": data.get("top_tier_count", 0),
            "top": [(c.get("name"), c.get("hit_count")) for c in candidates[:3]],
        }
        if data.get("kind") == "question":
            st.session_state.triage_thread_id = data["thread_id"]
            st.session_state.triage_messages.append(
                ("assistant", data.get("question") or "")
            )
        else:
            advice = data.get("final_advice") or {}
            label = advice.get("label", "建议")
            suggestion = advice.get("suggestion", "")
            message = f"【{label}】{suggestion}"
            if advice.get("level") != "no_symptom":
                message += "\n\n（本轮已结束，可继续描述新的症状开始新一轮）"
            st.session_state.triage_messages.append(
                ("assistant", message)
            )
            st.session_state.triage_thread_id = None  # 本轮建议已给出，下次重新开始
    except requests.RequestException as exc:
        st.session_state.triage_messages.append(
            ("assistant", f"请求失败（请确认 FastAPI 已在 8000 端口启动）：{exc}")
        )
    st.rerun()


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
