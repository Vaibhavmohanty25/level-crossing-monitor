"""Show written history answers with optional supporting records."""

import sqlite3
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from ollama import ResponseError

from crossing_monitor.dashboard_data import records_to_csv
from crossing_monitor.history import ask_history


DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "runtime"
    / "incidents.db"
)


def display_time(value):
    if not value:
        return "—"

    return (
        datetime.fromisoformat(value)
        .astimezone()
        .strftime("%d %b %Y, %H:%M:%S")
    )


def display_answer(result, key):
    # The written answer appears first.
    st.write(result["answer"])

    plan = result["filters"]
    duration = []

    if plan.get("minimum_duration") is not None:
        operator = (
            ">"
            if plan["minimum_exclusive"]
            else ">="
        )

        duration.append(
            f'{operator} {plan["minimum_duration"]}s'
        )

    if plan.get("maximum_duration") is not None:
        operator = (
            "<"
            if plan["maximum_exclusive"]
            else "<="
        )

        duration.append(
            f'{operator} {plan["maximum_duration"]}s'
        )

    with st.expander("Interpreted search"):
        st.write({
            "Operation": plan["action"],
            "Start date": (
                plan.get("start_date") or "Any"
            ),
            "End date": (
                plan.get("end_date") or "Any"
            ),
            "Highest recorded level": (
                plan.get("level") or "All"
            ),
            "Record status": plan.get(
                "status",
                "all",
            ),
            "Source": (
                plan.get("source") or "All"
            ),
            "Duration": (
                "; ".join(duration) or "Any"
            ),
        })

        st.caption(
            "Dates are inclusive local calendar days. "
            "Check these filters if an answer looks unexpected."
        )

    records = result["records"]

    if records:
        with st.expander(
            f"Supporting records ({len(records)})",
            expanded=False,
        ):
            table_rows = [
                {
                    "Incident ID": row["incident_id"],
                    "Recorded at": display_time(
                        row["started_at_utc"]
                    ),
                    "Duration (s)": round(
                        row["duration_seconds"],
                        3,
                    ),
                    "Highest level": row["max_level"],
                    "Status": row["status"],
                    "Source": row["source"],
                }
                for row in records
            ]

            st.dataframe(
                table_rows,
                hide_index=True,
                width="stretch",
            )

            st.download_button(
                "Download shown records",
                records_to_csv(records),
                file_name="history_answer_records.csv",
                mime="text/csv",
                key=f"history_csv_{key}",
            )

    if result.get("queried_at_utc"):
        st.caption(
            "Answer reflects database records at "
            f'{display_time(result["queried_at_utc"])}.'
        )


def main():
    st.title("Ask incident history")

    st.caption(
        "Ask a question and get a written answer. "
        "Expand supporting records only when "
        "you want to inspect the evidence."
    )

    st.write(
        "Try: **Tell me about yesterday's incidents** · "
        "**Which incident lasted more than 8 seconds?** · "
        "**Which incident lasted longest?**"
    )

    st.info(
        "Ask each question independently. Dates use "
        "this PC's local timezone and the incident's "
        "recorded start time. Counts represent tracking "
        "episodes; active durations may increase."
    )

    st.sidebar.header("History assistant")

    if st.sidebar.button("Clear conversation"):
        st.session_state["history_answers"] = []
        st.rerun()

    if not DATABASE_PATH.is_file():
        st.info(
            "Record an incident before using history questions."
        )
        return

    # Remove cached answers produced by the old version.
    if st.session_state.get("history_version") != 2:
        st.session_state["history_answers"] = []
        st.session_state["history_version"] = 2

    for index, item in enumerate(
        st.session_state["history_answers"]
    ):
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            display_answer(
                item["result"],
                index,
            )

    question = st.chat_input(
        "Ask about recorded incidents…",
        max_chars=1000,
    )

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Checking incident history…",
                    show_time=True,
                ):
                    result = ask_history(
                        DATABASE_PATH,
                        question,
                    )

            except (
                ResponseError,
                httpx.HTTPError,
                ConnectionError,
                OSError,
                ValueError,
                sqlite3.Error,
            ) as error:
                st.error(
                    f"Could not answer: {error}"
                )

            else:
                index = len(
                    st.session_state["history_answers"]
                )

                display_answer(
                    result,
                    index,
                )

                st.session_state[
                    "history_answers"
                ].append({
                    "question": question,
                    "result": result,
                })