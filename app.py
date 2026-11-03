"""Streamlit dashboard for crossing incidents."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from ollama import Client, ResponseError

from crossing_monitor.dashboard_data import (
    read_incidents,
    read_details,
    filter_incidents,
    records_to_csv,
)
from crossing_monitor.storage import IncidentStore
from crossing_monitor.history_page import main as history_page
from crossing_monitor.heatmap_page import main as heatmap_page
from crossing_monitor.ollama_analysis import (
    analyse_incident,
    ensure_table,
    save_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent

DATABASE_PATH = (
    PROJECT_ROOT / "data" / "runtime" / "incidents.db"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT / "data" / "runtime" / "reports"
)

MODEL = "qwen3-vl:2b-instruct"

st.set_page_config(
    page_title="Crossing Incident Review",
    page_icon="🚦",
    layout="wide",
)


def local_time(value):
    if not value:
        return "—"

    try:
        return (
            datetime.fromisoformat(value)
            .astimezone()
            .strftime("%d %b %Y, %H:%M:%S")
        )

    except (ValueError, TypeError):
        return str(value)


def generate_report(incident_id):
    store = IncidentStore(DATABASE_PATH)

    try:
        ensure_table(store)
        event = store.get(incident_id)

        if event["status"] != "closed":
            raise ValueError(
                "Close the incident before generating a report."
            )

        client = Client(
            host="http://localhost:11434",
            timeout=600,
        )

        report = analyse_incident(
            client,
            event,
            MODEL,
        )

        save_report(
            store,
            report,
            REPORT_DIRECTORY,
        )

    finally:
        store.close()


def main():
    st.title("Crossing incident review")

    st.caption(
        "Saved incidents, visual evidence, "
        "and local AI analysis"
    )

    st.sidebar.header("Filters")
    st.sidebar.button("Refresh records")

    st.sidebar.caption(
        "Press Refresh to load new monitoring records."
    )

    if not DATABASE_PATH.is_file():
        st.info(
            "Record an intrusion with the monitoring "
            "program first."
        )
        return

    try:
        records = read_incidents(DATABASE_PATH)

    except (
        OSError,
        sqlite3.Error,
        ValueError,
    ) as error:
        st.error(
            f"Could not read incident history: {error}"
        )
        return

    if not records:
        st.info(
            "No incidents recorded yet. Close the crossing "
            "and move a vehicle into it."
        )
        return

    levels = st.sidebar.multiselect(
        "Highest escalation level",
        ["ENTRY", "PERSISTENT", "PROLONGED"],
        default=["ENTRY", "PERSISTENT", "PROLONGED"],
    )

    status = st.sidebar.selectbox(
        "Record status",
        ["All", "closed", "active"],
    )

    source = st.sidebar.selectbox(
        "Video source",
        ["All"] + sorted({
            row["source"]
            for row in records
        }),
    )

    minimum = st.sidebar.number_input(
        "Minimum duration (seconds)",
        min_value=0.0,
        value=0.0,
        step=0.5,
    )

    search = (
        st.sidebar.text_input("Search incident ID")
        .strip()
        .lower()
    )

    filtered = filter_incidents(
        records,
        levels,
        None if status == "All" else status,
        None if source == "All" else source,
        minimum,
    )

    if search:
        filtered = [
            row
            for row in filtered
            if search in row["incident_id"].lower()
        ]

    columns = st.columns(4)

    columns[0].metric(
        "Matching incidents",
        len(filtered),
    )

    columns[1].metric(
        "Prolonged episodes",
        sum(
            row["max_level"] == "PROLONGED"
            for row in filtered
        ),
    )

    columns[2].metric(
        "Closed records",
        sum(
            row["status"] == "closed"
            for row in filtered
        ),
    )

    average = (
        sum(
            row["duration_seconds"]
            for row in filtered
        ) / len(filtered)
        if filtered
        else 0
    )

    columns[3].metric(
        "Mean recorded duration",
        f"{average:.1f}s",
    )

    st.caption(
        "Filters apply to the latest 1,000 records. "
        "Counts represent tracking episodes. "
        "Times use this PC's local timezone."
    )

    if not filtered:
        st.info("No incidents match these filters.")
        return

    st.subheader("Incident history")

    st.dataframe(
        [
            {
                "Incident": row["incident_id"][:8],
                "Track": row["track_id"],
                "Started": local_time(
                    row["started_at_utc"]
                ),
                "Duration (s)": round(
                    row["duration_seconds"],
                    2,
                ),
                "Highest level": row["max_level"],
                "Status": row["status"],
                "End reason": (
                    row["end_reason"] or "ongoing"
                ),
                "Source": row["source"],
            }
            for row in filtered
        ],
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "Download filtered history",
        records_to_csv(filtered),
        file_name="filtered_incidents.csv",
        mime="text/csv",
    )

    lookup = {
        row["incident_id"]: row
        for row in filtered
    }

    selected = st.selectbox(
        "Choose an incident to review",
        list(lookup),
        format_func=lambda key: (
            f'{key[:8]} • '
            f'Track {lookup[key]["track_id"]} • '
            f'{lookup[key]["max_level"]} • '
            f'{lookup[key]["duration_seconds"]:.1f}s'
        ),
    )

    try:
        event, report = read_details(
            DATABASE_PATH,
            selected,
        )

    except (
        OSError,
        sqlite3.Error,
        ValueError,
        KeyError,
    ) as error:
        st.error(
            f"Could not load this incident: {error}"
        )
        return

    details_tab, evidence_tab, report_tab = st.tabs([
        "Recorded facts",
        "Screenshots",
        "Ollama report",
    ])

    with details_tab:
        st.write({
            "Incident ID": event["incident_id"],
            "Tracking ID": event["track_id"],
            "Source": event["source"],
            "Started": local_time(
                event["started_at_utc"]
            ),
            "Record closed": local_time(
                event["ended_at_utc"]
            ),
            "Recorded duration (s)": round(
                event["duration_seconds"],
                3,
            ),
            "Highest level": event["max_level"],
            "Record status": event["status"],
            "End reason": (
                event["end_reason"] or "ongoing"
            ),
        })

        st.caption(
            "Opening the crossing, losing a track, or "
            "stopping monitoring ends a record without "
            "proving the car physically left."
        )

    with evidence_tab:
        snapshots = event["snapshots"]

        if not snapshots:
            st.info(
                "No screenshots were saved for this incident."
            )

        else:
            image_columns = st.columns(
                min(3, len(snapshots))
            )

            for index, snapshot in enumerate(snapshots):
                with image_columns[
                    index % len(image_columns)
                ]:
                    st.write(snapshot["level"])

                    path = Path(snapshot["path"])

                    if path.is_file():
                        st.image(
                            str(path),
                            width="stretch",
                            caption=local_time(
                                snapshot["recorded_at_utc"]
                            ),
                        )

                    else:
                        st.warning(
                            "The saved screenshot "
                            "is missing from disk."
                        )

    with report_tab:
        if report:
            st.caption(
                f'Model: {report.get("model", "unknown")} • '
                f'Generated: '
                f'{local_time(report.get("analysed_at_utc"))}'
            )

            st.subheader("Database summary")

            st.write(
                report.get(
                    "record_summary",
                    "No summary saved.",
                )
            )

            visual = report.get(
                "ai_visual_analysis",
                {},
            )

            st.subheader("AI visual description")

            st.write(
                visual.get(
                    "scene_description",
                    "No description saved.",
                )
            )

            st.write("**Visible observations**")

            for observation in visual.get(
                "visible_observations",
                [],
            ):
                st.write(f"• {observation}")

            st.write("**Uncertainties**")

            uncertainties = visual.get(
                "uncertainties",
                [],
            )

            if uncertainties:
                for uncertainty in uncertainties:
                    st.write(f"• {uncertainty}")

            else:
                st.caption(
                    "The model listed no uncertainties; "
                    "this does not establish accuracy."
                )

            st.download_button(
                "Download analysis JSON",
                json.dumps(report, indent=2),
                file_name=(
                    f"{selected}_analysis.json"
                ),
                mime="application/json",
            )

        else:
            st.info(
                "No Ollama analysis has been saved "
                "for this incident."
            )

        st.caption(
            f"Analysis uses {MODEL}. Keep Ollama running. "
            "Existing reports are replaced when regenerated."
        )

        if st.button(
            (
                "Regenerate analysis"
                if report
                else "Generate analysis"
            ),
            disabled=event["status"] != "closed",
        ):
            try:
                with st.spinner(
                    "Analysing the saved screenshot…",
                    show_time=True,
                ):
                    generate_report(selected)

            except (
                ResponseError,
                httpx.HTTPError,
                ConnectionError,
                OSError,
                ValueError,
                sqlite3.Error,
            ) as error:
                st.error(
                    "Analysis failed; "
                    f"original incident retained: {error}"
                )

            else:
                st.rerun()

        if event["status"] != "closed":
            st.caption(
                "End the active incident before "
                "generating its report."
            )


st.navigation(
    [
        st.Page(
            main,
            title="Incident review",
            default=True,
        ),
        st.Page(
            history_page,
            title="Ask incident history",
            url_path="history",
        ),
        st.Page(
            heatmap_page,
            title="Intrusion heatmap",
            url_path="heatmap",
        ),
    ],
    position="sidebar",
    expanded=True,
).run()