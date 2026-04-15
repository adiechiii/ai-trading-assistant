import base64
import json
import os
import traceback
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📈",
    layout="wide",
)

HISTORY_FILE = "history.json"
REQUEST_TIMEOUT = 20

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

OPENAI_MODEL = "gpt-5.4-mini"
CLAUDE_MODEL = "claude-sonnet-4-6"

# -----------------------------
# Helpers
# -----------------------------
def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict)]


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_confidence(value: Any) -> float:
    num = safe_float(value)
    if num is None:
        return 0.0
    if 0 <= num <= 1:
        return round(num * 100, 2)
    return round(num, 2)


def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def format_number(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.5f}".rstrip("0").rstrip(".")
    return str(value)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = (raw_text or "").strip()

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw_text[start:end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise json.JSONDecodeError("No JSON object found", raw_text, 0)


def file_to_data_url(uploaded_file) -> str:
    mime_type = getattr(uploaded_file, "type", None) or "image/png"
    file_bytes = uploaded_file.getvalue()
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# -----------------------------
# History
# -----------------------------
def ensure_history_file() -> None:
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)


def load_history() -> list[dict[str, Any]]:
    ensure_history_file()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return safe_list_of_dicts(data)
    except (json.JSONDecodeError, OSError):
        return []


def save_history(items: list[dict[str, Any]]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(safe_list_of_dicts(items), f, indent=2, ensure_ascii=False)


def append_history(record: dict[str, Any], max_items: int = 100) -> None:
    history = load_history()
    history.insert(0, safe_dict(record))
    history = history[:max_items]
    save_history(history)


# -----------------------------
# Empty schemas
# -----------------------------
def get_empty_analysis_extraction() -> dict[str, Any]:
    return {
        "title": None,
        "summary": None,
        "symbol": None,
        "asset_type": None,
        "bias": None,
        "mentioned_entry": None,
        "mentioned_stop_loss": None,
        "mentioned_target_1": None,
        "mentioned_target_2": None,
        "key_levels_text": None,
        "reason": "Analysis screenshot extraction unavailable.",
        "confidence": 0,
    }


def get_empty_chart_analysis() -> dict[str, Any]:
    return {
        "timeframe": "1H",
        "symbol": None,
        "asset_type": None,
        "trend": None,
        "structure": None,
        "position": "unclear",
        "support_zone": None,
        "resistance_zone": None,
        "is_tradable_now": False,
        "reason": "Chart analysis unavailable.",
        "confidence": 0,
        "summary": "Chart analysis unavailable.",
    }


def get_empty_final_model_analysis(symbol: str | None, asset_type: str | None) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "asset_type": asset_type,
        "bias": None,
        "setup": {
            "type": None,
            "market_structure": None,
            "quality_score": None,
        },
        "levels": {
            "support": None,
            "resistance": None,
            "breakout_level": None,
            "invalidation_level": None,
        },
        "trade_plan": {
            "entry_zone": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "risk_reward_comment": None,
        },
        "live_context": {
            "current_price": None,
            "position_vs_setup": "unclear",
        },
        "final_judgement": {
            "is_tradable": False,
            "reason": "Model analysis unavailable.",
            "confidence": 0,
        },
        "trader_summary": "Model analysis unavailable.",
    }


# -----------------------------
# Vision prompts
# -----------------------------
def build_analysis_image_prompt() -> str:
    return """
You are reading a screenshot of a market analysis article or post.

Your job:
- extract the trading idea from the screenshot
- identify the symbol if possible
- identify asset type if possible
- identify bias if possible
- extract any mentioned entry, stop loss, target 1, target 2 if present
- summarize the idea conservatively
- if values are missing or unclear, return null
- do not invent anything
- confidence must be 0 to 100

Return exactly one JSON object:

{
  "title": string | null,
  "summary": string | null,
  "symbol": string | null,
  "asset_type": "crypto" | "forex" | "commodity" | "stock" | "index" | null,
  "bias": "LONG" | "SHORT" | "NEUTRAL" | null,
  "mentioned_entry": string | null,
  "mentioned_stop_loss": number | null,
  "mentioned_target_1": number | null,
  "mentioned_target_2": number | null,
  "key_levels_text": string | null,
  "reason": string,
  "confidence": number
}
""".strip()


def build_chart_prompt() -> str:
    return """
You are a professional trader reading a TradingView chart screenshot.

Assume the screenshot is 1H unless clearly shown otherwise.

Your job:
- identify symbol if visible
- identify asset type if clear
- identify trend
- identify structure
- judge whether the setup is tradable right now
- classify current position as near_entry / extended / late / invalid / unclear
- approximate visible support/resistance zones if possible
- be conservative
- confidence must be 0 to 100

Return exactly one JSON object:

{
  "timeframe": string | null,
  "symbol": string | null,
  "asset_type": "crypto" | "forex" | "commodity" | "stock" | "index" | null,
  "trend": "bullish" | "bearish" | "range" | "unclear" | null,
  "structure": string | null,
  "position": "near_entry" | "extended" | "late" | "invalid" | "unclear",
  "support_zone": string | null,
  "resistance_zone": string | null,
  "is_tradable_now": true | false,
  "reason": string,
  "confidence": number,
  "summary": string
}
""".strip()


def build_openai_trader_prompt(analysis_data: dict[str, Any], chart_data: dict[str, Any]) -> str:
    return f"""
You are a professional trader.

You have two sources:
1. Analysis screenshot extraction
2. TradingView chart screenshot analysis

Your task:
- combine both
- focus on execution quality NOW
- do not summarize only
- decide if the trade is tradable now
- be conservative
- if information is missing, return null
- if setup is late, extended, invalid, or weak, say so
- confidence must be 0 to 100

ANALYSIS SCREENSHOT DATA:
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

CHART SCREENSHOT DATA:
{json.dumps(chart_data, ensure_ascii=False, indent=2)}

Return exactly one JSON object:

{{
  "symbol": string | null,
  "asset_type": "crypto" | "forex" | "commodity" | "stock" | "index" | null,
  "bias": "LONG" | "SHORT" | "NEUTRAL" | null,
  "setup": {{
    "type": string | null,
    "market_structure": string | null,
    "quality_score": number | null
  }},
  "levels": {{
    "support": number | null,
    "resistance": number | null,
    "breakout_level": number | null,
    "invalidation_level": number | null
  }},
  "trade_plan": {{
    "entry_zone": string | null,
    "stop_loss": number | null,
    "target_1": number | null,
    "target_2": number | null,
    "risk_reward_comment": string | null
  }},
  "live_context": {{
    "current_price": number | null,
    "position_vs_setup": "near_entry" | "extended" | "late" | "invalid" | "unclear"
  }},
  "final_judgement": {{
    "is_tradable": true | false,
    "reason": string,
    "confidence": number
  }},
  "trader_summary": string
}}
""".strip()


def build_claude_risk_prompt(analysis_data: dict[str, Any], chart_data: dict[str, Any], openai_data: dict[str, Any]) -> str:
    return f"""
You are a strict risk manager.

You have:
1. Analysis screenshot extraction
2. Chart screenshot analysis
3. OpenAI trader view

Your job:
- challenge the trade
- focus on risk, execution quality, missing data, bad risk/reward, late entries
- be stricter than the trader
- if unclear, return null
- confidence must be 0 to 100

ANALYSIS SCREENSHOT DATA:
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

CHART SCREENSHOT DATA:
{json.dumps(chart_data, ensure_ascii=False, indent=2)}

OPENAI TRADER VIEW:
{json.dumps(openai_data, ensure_ascii=False, indent=2)}

Return exactly one JSON object:

{{
  "symbol": string | null,
  "asset_type": "crypto" | "forex" | "commodity" | "stock" | "index" | null,
  "bias": "LONG" | "SHORT" | "NEUTRAL" | null,
  "setup": {{
    "type": string | null,
    "market_structure": string | null,
    "quality_score": number | null
  }},
  "levels": {{
    "support": number | null,
    "resistance": number | null,
    "breakout_level": number | null,
    "invalidation_level": number | null
  }},
  "trade_plan": {{
    "entry_zone": string | null,
    "stop_loss": number | null,
    "target_1": number | null,
    "target_2": number | null,
    "risk_reward_comment": string | null
  }},
  "live_context": {{
    "current_price": number | null,
    "position_vs_setup": "near_entry" | "extended" | "late" | "invalid" | "unclear"
  }},
  "final_judgement": {{
    "is_tradable": true | false,
    "reason": string,
    "confidence": number
  }},
  "trader_summary": string
}}
""".strip()


# -----------------------------
# OpenAI vision
# -----------------------------
def analyze_image_to_json(uploaded_file, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    if not uploaded_file:
        return {
            "success": False,
            "data": fallback,
            "error": "No image uploaded.",
            "raw_response": None,
        }

    if not OPENAI_API_KEY:
        return {
            "success": False,
            "data": fallback,
            "error": "Missing OPENAI_API_KEY in .env",
            "raw_response": None,
        }

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        data_url = file_to_data_url(uploaded_file)

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )

        raw_text = (getattr(response, "output_text", None) or "").strip()
        if not raw_text:
            return {
                "success": False,
                "data": fallback,
                "error": "Model returned empty output.",
                "raw_response": None,
            }

        parsed = extract_json_object(raw_text)

        return {
            "success": True,
            "data": parsed if isinstance(parsed, dict) else fallback,
            "error": None,
            "raw_response": raw_text,
        }

    except Exception as e:
        return {
            "success": False,
            "data": fallback,
            "error": f"Image analysis failed: {e}",
            "raw_response": None,
        }


# -----------------------------
# Text model calls
# -----------------------------
def analyze_combined_with_openai(analysis_data: dict[str, Any], chart_data: dict[str, Any]) -> dict[str, Any]:
    fallback_symbol = analysis_data.get("symbol") or chart_data.get("symbol")
    fallback_asset_type = analysis_data.get("asset_type") or chart_data.get("asset_type")
    empty = get_empty_final_model_analysis(fallback_symbol, fallback_asset_type)

    if not OPENAI_API_KEY:
        return {
            "success": False,
            "analysis": empty,
            "error": "Missing OPENAI_API_KEY in .env",
            "raw_response": None,
        }

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = build_openai_trader_prompt(analysis_data, chart_data)

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )

        raw_text = (getattr(response, "output_text", None) or "").strip()
        if not raw_text:
            return {
                "success": False,
                "analysis": empty,
                "error": "OpenAI returned empty output.",
                "raw_response": None,
            }

        parsed = extract_json_object(raw_text)

        return {
            "success": True,
            "analysis": parsed if isinstance(parsed, dict) else empty,
            "error": None,
            "raw_response": raw_text,
        }

    except Exception as e:
        return {
            "success": False,
            "analysis": empty,
            "error": f"OpenAI combined analysis failed: {e}",
            "raw_response": None,
        }


def extract_claude_text(response: Any) -> str:
    parts = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def analyze_combined_with_claude(
    analysis_data: dict[str, Any],
    chart_data: dict[str, Any],
    openai_data: dict[str, Any],
) -> dict[str, Any]:
    fallback_symbol = analysis_data.get("symbol") or chart_data.get("symbol")
    fallback_asset_type = analysis_data.get("asset_type") or chart_data.get("asset_type")
    empty = get_empty_final_model_analysis(fallback_symbol, fallback_asset_type)

    if not ANTHROPIC_API_KEY:
        return {
            "success": False,
            "analysis": empty,
            "error": "Missing ANTHROPIC_API_KEY in .env",
            "raw_response": None,
        }

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = build_claude_risk_prompt(analysis_data, chart_data, openai_data)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1800,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = extract_claude_text(response)
        if not raw_text:
            return {
                "success": False,
                "analysis": empty,
                "error": "Claude returned empty output.",
                "raw_response": None,
            }

        parsed = extract_json_object(raw_text)

        return {
            "success": True,
            "analysis": parsed if isinstance(parsed, dict) else empty,
            "error": None,
            "raw_response": raw_text,
        }

    except Exception as e:
        return {
            "success": False,
            "analysis": empty,
            "error": f"Claude combined analysis failed: {e}",
            "raw_response": None,
        }


# -----------------------------
# Decision engine
# -----------------------------
def compute_decision_label(openai_analysis: dict[str, Any], claude_analysis: dict[str, Any], chart_data: dict[str, Any]) -> str:
    openai_analysis = safe_dict(openai_analysis)
    claude_analysis = safe_dict(claude_analysis)
    chart_data = safe_dict(chart_data)

    openai_final = safe_dict(openai_analysis.get("final_judgement"))
    claude_final = safe_dict(claude_analysis.get("final_judgement"))

    openai_setup = safe_dict(openai_analysis.get("setup"))
    claude_setup = safe_dict(claude_analysis.get("setup"))

    openai_live = safe_dict(openai_analysis.get("live_context"))
    claude_live = safe_dict(claude_analysis.get("live_context"))

    o_tradable = bool(openai_final.get("is_tradable"))
    c_tradable = bool(claude_final.get("is_tradable"))

    o_conf = normalize_confidence(openai_final.get("confidence"))
    c_conf = normalize_confidence(claude_final.get("confidence"))

    o_quality = safe_float(openai_setup.get("quality_score")) or 0
    c_quality = safe_float(claude_setup.get("quality_score")) or 0

    o_pos = (openai_live.get("position_vs_setup") or "").lower()
    c_pos = (claude_live.get("position_vs_setup") or "").lower()
    chart_pos = (chart_data.get("position") or "").lower()

    avg_conf = (o_conf + c_conf) / 2
    avg_quality = (o_quality + c_quality) / 2

    if o_pos in {"invalid"} or c_pos in {"invalid"} or chart_pos == "invalid":
        return "NOT ACTIONABLE"

    if chart_pos in {"late", "extended"} and not chart_data.get("is_tradable_now", False):
        return "NOT ACTIONABLE"

    if o_pos in {"late", "extended"} and c_pos in {"late", "extended"}:
        return "NOT ACTIONABLE"

    if o_tradable and c_tradable and avg_conf >= 75 and avg_quality >= 70:
        return "STRONG"

    if o_tradable or c_tradable:
        if chart_pos == "near_entry" and chart_data.get("is_tradable_now"):
            return "WEAK"
        if avg_conf >= 60:
            return "WEAK"
        return "WATCHLIST"

    if chart_pos == "near_entry":
        return "WATCHLIST"

    return "NOT ACTIONABLE"


def build_final_decision(
    analysis_data: dict[str, Any],
    chart_data: dict[str, Any],
    openai_analysis: dict[str, Any],
    claude_analysis: dict[str, Any],
) -> dict[str, Any]:
    label = compute_decision_label(openai_analysis, claude_analysis, chart_data)

    openai_final = safe_dict(openai_analysis.get("final_judgement"))
    claude_final = safe_dict(claude_analysis.get("final_judgement"))

    openai_trade = safe_dict(openai_analysis.get("trade_plan"))
    claude_trade = safe_dict(claude_analysis.get("trade_plan"))

    symbol = (
        openai_analysis.get("symbol")
        or claude_analysis.get("symbol")
        or analysis_data.get("symbol")
        or chart_data.get("symbol")
    )
    asset_type = (
        openai_analysis.get("asset_type")
        or claude_analysis.get("asset_type")
        or analysis_data.get("asset_type")
        or chart_data.get("asset_type")
    )

    avg_conf = round(
        (
            normalize_confidence(openai_final.get("confidence"))
            + normalize_confidence(claude_final.get("confidence"))
            + normalize_confidence(chart_data.get("confidence"))
        ) / 3,
        1,
    )

    if label == "STRONG":
        reason = "Both models and chart alignment suggest a tradable setup with acceptable execution quality."
    elif label == "WEAK":
        reason = "There may be a setup, but conviction or execution quality is not strong enough for aggressive execution."
    elif label == "WATCHLIST":
        reason = "Setup may become tradable, but confirmation or timing is still not strong enough."
    else:
        reason = "Current execution is poor, late, weak, or unclear. Better to stand aside."

    return {
        "decision": label,
        "symbol": symbol,
        "asset_type": asset_type,
        "entry_zone": openai_trade.get("entry_zone") or claude_trade.get("entry_zone") or analysis_data.get("mentioned_entry"),
        "stop_loss": claude_trade.get("stop_loss") or openai_trade.get("stop_loss") or analysis_data.get("mentioned_stop_loss"),
        "target_1": openai_trade.get("target_1") or claude_trade.get("target_1") or analysis_data.get("mentioned_target_1"),
        "target_2": openai_trade.get("target_2") or claude_trade.get("target_2") or analysis_data.get("mentioned_target_2"),
        "average_confidence": avg_conf,
        "reason": reason,
        "combined_summary": (
            f"Analysis screenshot: {analysis_data.get('summary')} | "
            f"Chart: {chart_data.get('summary')} | "
            f"OpenAI: {openai_final.get('reason')} | "
            f"Claude: {claude_final.get('reason')}"
        ),
    }


# -----------------------------
# Telegram
# -----------------------------
def decision_emoji(decision: str) -> str:
    mapping = {
        "STRONG": "🟢",
        "WEAK": "🟡",
        "WATCHLIST": "👀",
        "NOT ACTIONABLE": "⛔",
    }
    return mapping.get(decision, "📈")


def format_telegram_message(
    analysis_data: dict[str, Any],
    chart_data: dict[str, Any],
    final_decision: dict[str, Any],
    openai_analysis: dict[str, Any],
    claude_analysis: dict[str, Any],
) -> str:
    emoji = decision_emoji(final_decision.get("decision"))
    openai_final = safe_dict(openai_analysis.get("final_judgement"))
    claude_final = safe_dict(claude_analysis.get("final_judgement"))
    openai_live = safe_dict(openai_analysis.get("live_context"))
    claude_live = safe_dict(claude_analysis.get("live_context"))

    return (
        f"{emoji} AI Trading Assistant Alert\n\n"
        f"Title: {analysis_data.get('title')}\n"
        f"Symbol: {final_decision.get('symbol')}\n"
        f"Asset Type: {final_decision.get('asset_type')}\n"
        f"Decision: {final_decision.get('decision')}\n"
        f"Entry: {final_decision.get('entry_zone') or 'null'}\n"
        f"SL: {format_number(final_decision.get('stop_loss'))}\n"
        f"TP1: {format_number(final_decision.get('target_1'))}\n"
        f"TP2: {format_number(final_decision.get('target_2'))}\n"
        f"Avg Confidence: {final_decision.get('average_confidence')}\n\n"
        f"Final Reason: {final_decision.get('reason')}\n\n"
        f"Chart Trend: {chart_data.get('trend')} | "
        f"Chart Pos: {chart_data.get('position')} | "
        f"Chart Tradable: {chart_data.get('is_tradable_now')}\n\n"
        f"OpenAI Bias: {openai_analysis.get('bias')} | Tradable: {openai_final.get('is_tradable')} | Pos: {openai_live.get('position_vs_setup')}\n"
        f"Claude Bias: {claude_analysis.get('bias')} | Tradable: {claude_final.get('is_tradable')} | Pos: {claude_live.get('position_vs_setup')}\n"
    )


def send_to_telegram(message: str) -> dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {
            "success": False,
            "error": "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env",
        }

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("ok"):
            return {"success": True, "error": None}

        return {
            "success": False,
            "error": data.get("description", "Telegram API error"),
        }

    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Telegram request failed: {e}"}


# -----------------------------
# UI renderers
# -----------------------------
def render_analysis_extraction(result: dict[str, Any]) -> None:
    data = safe_dict(result.get("data"))
    st.subheader("Analysis Screenshot Extraction")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Title", data.get("title") or "null")
    c2.metric("Symbol", data.get("symbol") or "null")
    c3.metric("Asset Type", data.get("asset_type") or "null")
    c4.metric("Bias", data.get("bias") or "null")

    st.markdown(f"**Summary:** {data.get('summary') or 'null'}")
    st.markdown(f"**Mentioned Entry:** {data.get('mentioned_entry') or 'null'}")
    st.markdown(f"**Mentioned SL:** {format_number(data.get('mentioned_stop_loss'))}")
    st.markdown(f"**Mentioned TP1:** {format_number(data.get('mentioned_target_1'))}")
    st.markdown(f"**Mentioned TP2:** {format_number(data.get('mentioned_target_2'))}")
    st.markdown(f"**Levels Text:** {data.get('key_levels_text') or 'null'}")
    st.markdown(f"**Reason:** {data.get('reason') or 'null'}")

    if result.get("raw_response"):
        with st.expander("Analysis Screenshot Raw Response"):
            st.code(result["raw_response"], language="json")

    if not result.get("success"):
        st.warning(result.get("error"))


def render_chart_result(result: dict[str, Any]) -> None:
    data = safe_dict(result.get("data"))
    st.subheader("Chart Screenshot Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Timeframe", data.get("timeframe") or "null")
    c2.metric("Trend", data.get("trend") or "null")
    c3.metric("Position", data.get("position") or "null")
    c4.metric("Tradable Now", "YES" if data.get("is_tradable_now") else "NO")

    st.markdown(f"**Structure:** {data.get('structure') or 'null'}")
    st.markdown(f"**Support Zone:** {data.get('support_zone') or 'null'}")
    st.markdown(f"**Resistance Zone:** {data.get('resistance_zone') or 'null'}")
    st.markdown(f"**Reason:** {data.get('reason') or 'null'}")
    st.markdown(f"**Summary:** {data.get('summary') or 'null'}")

    if result.get("raw_response"):
        with st.expander("Chart Screenshot Raw Response"):
            st.code(result["raw_response"], language="json")

    if not result.get("success"):
        st.warning(result.get("error"))


def render_model_analysis(title: str, result: dict[str, Any]) -> None:
    analysis = safe_dict(result.get("analysis"))
    final_judgement = safe_dict(analysis.get("final_judgement"))
    setup = safe_dict(analysis.get("setup"))
    live_context = safe_dict(analysis.get("live_context"))

    st.markdown(f"### {title}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", analysis.get("symbol") or "null")
    c2.metric("Bias", analysis.get("bias") or "null")
    c3.metric("Tradable", "YES" if final_judgement.get("is_tradable") else "NO")
    c4.metric("Confidence", format_number(normalize_confidence(final_judgement.get("confidence"))))

    st.markdown(f"**Reason:** {final_judgement.get('reason') or 'null'}")
    st.markdown(f"**Setup Type:** {setup.get('type') or 'null'}")
    st.markdown(f"**Market Structure:** {setup.get('market_structure') or 'null'}")
    st.markdown(f"**Position vs Setup:** {live_context.get('position_vs_setup') or 'null'}")
    st.markdown(f"**Summary:** {analysis.get('trader_summary') or 'null'}")

    with st.expander(f"{title} Parsed JSON"):
        st.json(analysis)

    if result.get("raw_response"):
        with st.expander(f"{title} Raw Response"):
            st.code(result["raw_response"], language="json")

    if not result.get("success"):
        st.warning(result.get("error"))


def render_final_decision(final_decision: dict[str, Any]) -> None:
    st.subheader("Final Trader Decision")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision", final_decision.get("decision"))
    c2.metric("Symbol", final_decision.get("symbol") or "null")
    c3.metric("Asset Type", final_decision.get("asset_type") or "null")
    c4.metric("Avg Confidence", format_number(final_decision.get("average_confidence")))

    st.markdown("### Execution View")
    st.info(final_decision.get("reason"))

    st.markdown("### Final Setup")
    s1, s2, s3, s4 = st.columns(4)
    s1.text_input("Entry", value=final_decision.get("entry_zone") or "null", disabled=True)
    s2.text_input("Stop Loss", value=format_number(final_decision.get("stop_loss")), disabled=True)
    s3.text_input("Target 1", value=format_number(final_decision.get("target_1")), disabled=True)
    s4.text_input("Target 2", value=format_number(final_decision.get("target_2")), disabled=True)

    st.markdown("### Combined Summary")
    st.write(final_decision.get("combined_summary"))


def render_history_sidebar(history: list[dict[str, Any]], limit: int = 10) -> None:
    st.sidebar.title("Recent History")

    if st.sidebar.button("🗑 Clear History", use_container_width=True):
        save_history([])
        st.sidebar.success("History cleared.")
        st.rerun()

    st.sidebar.caption(f"{len(history)} saved items")

    if not history:
        st.sidebar.caption("No saved analyses yet.")
        return

    for item in history[:limit]:
        item = safe_dict(item)
        with st.sidebar.expander(f"{item.get('decision', 'UNKNOWN')} • {item.get('symbol', 'N/A')}"):
            st.write(f"**Time:** {item.get('timestamp_utc', 'N/A')}")
            st.write(f"**Title:** {item.get('title', 'N/A')}")
            st.write(f"**Avg Confidence:** {item.get('average_confidence', 'N/A')}")
            st.write(f"**Telegram:** {'Yes' if item.get('telegram_sent') else 'No'}")


def render_history_main(history: list[dict[str, Any]], limit: int = 20) -> None:
    st.subheader("Analysis History")

    if not history:
        st.caption("No history yet.")
        return

    for item in history[:limit]:
        item = safe_dict(item)
        with st.expander(f"{item.get('decision', 'UNKNOWN')} • {item.get('symbol', 'N/A')} • {item.get('timestamp_utc', 'N/A')}"):
            st.markdown(f"**Title:** {item.get('title', 'N/A')}")
            st.markdown(f"**Symbol:** {item.get('symbol', 'N/A')}")
            st.markdown(f"**Asset Type:** {item.get('asset_type', 'N/A')}")
            st.markdown(f"**Entry:** {item.get('entry_zone', 'null')}")
            st.markdown(f"**SL:** {format_number(item.get('stop_loss'))}")
            st.markdown(f"**TP1:** {format_number(item.get('target_1'))}")
            st.markdown(f"**TP2:** {format_number(item.get('target_2'))}")
            st.markdown(f"**Final Reason:** {item.get('final_reason', 'N/A')}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_history_file()
    history = load_history()

    st.title("📈 AI Trading Assistant")
    st.caption("Two-screenshot version: analysis screenshot + TradingView screenshot")

    render_history_sidebar(history)

    st.markdown(
        """
        Upload:
        - analysis screenshot
        - TradingView 1H screenshot

        Flow:
        - read analysis screenshot
        - read TradingView chart screenshot
        - combine both with OpenAI
        - review risk with Claude
        - build final decision
        - optionally send to Telegram
        - save to history
        """
    )

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        send_telegram = st.checkbox("Send to Telegram")
    with top_col2:
        st.write("")

    analysis_image = st.file_uploader(
        "Upload Analysis Screenshot",
        type=["png", "jpg", "jpeg"],
        key="analysis_image",
    )

    chart_image = st.file_uploader(
        "Upload TradingView Screenshot (1H preferred)",
        type=["png", "jpg", "jpeg"],
        key="chart_image",
    )

    if analysis_image:
        st.image(analysis_image, caption="Analysis screenshot", use_container_width=True)

    if chart_image:
        st.image(chart_image, caption="TradingView screenshot", use_container_width=True)

    analyze = st.button("Analyze", type="primary", use_container_width=True)

    if not analyze:
        render_history_main(history)
        return

    if not analysis_image:
        st.error("Please upload the analysis screenshot.")
        return

    if not chart_image:
        st.error("Please upload the TradingView screenshot.")
        return

    try:
        with st.spinner("Reading analysis screenshot..."):
            analysis_extract_result = analyze_image_to_json(
                analysis_image,
                build_analysis_image_prompt(),
                get_empty_analysis_extraction(),
            )

        with st.spinner("Reading TradingView chart screenshot..."):
            chart_result = analyze_image_to_json(
                chart_image,
                build_chart_prompt(),
                get_empty_chart_analysis(),
            )

        analysis_data = safe_dict(analysis_extract_result.get("data"))
        chart_data = safe_dict(chart_result.get("data"))

        with st.spinner("Building trader view with OpenAI..."):
            openai_result = analyze_combined_with_openai(analysis_data, chart_data)

        openai_analysis = safe_dict(openai_result.get("analysis"))

        with st.spinner("Building risk review with Claude..."):
            claude_result = analyze_combined_with_claude(analysis_data, chart_data, openai_analysis)

        claude_analysis = safe_dict(claude_result.get("analysis"))

        final_decision = build_final_decision(
            analysis_data=analysis_data,
            chart_data=chart_data,
            openai_analysis=openai_analysis,
            claude_analysis=claude_analysis,
        )

        telegram_result = {}
        telegram_sent = False

        if send_telegram:
            telegram_message = format_telegram_message(
                analysis_data=analysis_data,
                chart_data=chart_data,
                final_decision=final_decision,
                openai_analysis=openai_analysis,
                claude_analysis=claude_analysis,
            )
            with st.spinner("Sending to Telegram..."):
                telegram_result = send_to_telegram(telegram_message)
            telegram_sent = bool(telegram_result.get("success"))

        history_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "title": analysis_data.get("title"),
            "symbol": final_decision.get("symbol"),
            "asset_type": final_decision.get("asset_type"),
            "decision": final_decision.get("decision"),
            "average_confidence": final_decision.get("average_confidence"),
            "entry_zone": final_decision.get("entry_zone"),
            "stop_loss": final_decision.get("stop_loss"),
            "target_1": final_decision.get("target_1"),
            "target_2": final_decision.get("target_2"),
            "final_reason": final_decision.get("reason"),
            "telegram_sent": telegram_sent,
        }
        append_history(history_record)
        history = load_history()

        st.success("Screenshots processed successfully.")

        render_analysis_extraction(analysis_extract_result)
        render_chart_result(chart_result)
        render_final_decision(final_decision)

        col1, col2 = st.columns(2)
        with col1:
            render_model_analysis("OpenAI Trader View", openai_result)
        with col2:
            render_model_analysis("Claude Risk Manager", claude_result)

        if send_telegram:
            st.subheader("Telegram Status")
            if telegram_result.get("success"):
                st.success("Telegram message sent successfully.")
            else:
                st.warning(telegram_result.get("error") or "Telegram send failed.")

        st.divider()
        render_history_main(history)

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        with st.expander("Debug traceback", expanded=True):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()