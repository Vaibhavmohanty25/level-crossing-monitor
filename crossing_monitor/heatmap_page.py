"""Review session-specific intrusion occupancy heatmaps."""

import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

from crossing_monitor.heatmap import (
    list_sessions,
    read_heatmap,
    render_heatmap,
    cells_csv,
)


DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "runtime"
    / "incidents.db"
)

LEVELS = [
    "ENTRY",
    "PERSISTENT",
    "PROLONGED",
]


def local_time(value):
    return (
        datetime.fromisoformat(value)
        .astimezone()
        .strftime("%d %b %Y, %H:%M:%S")
    )


def main():
    st.title("Intrusion heatmap")

    st.caption(
        "Where detected vehicles spend time "
        "inside the closed crossing."
    )

    st.sidebar.button(
        "Refresh heatmaps",
        key="heatmap_refresh",
    )

    try:
        sessions = list_sessions(DATABASE_PATH)

    except (OSError, sqlite3.Error) as error:
        st.error(
            f"Could not read monitoring sessions: {error}"
        )
        return

    if not sessions:
        st.info(
            "Run monitoring with the heatmap update first. "
            "Older incidents have no position data."
        )
        return

    source = st.sidebar.selectbox(
        "Camera or video source",
        ["All"] + sorted({
            session["source"]
            for session in sessions
        }),
        key="heatmap_source",
    )

    choices = {
        session["session_id"]: session
        for session in sessions
        if (
            source == "All"
            or session["source"] == source
        )
    }

    selected = st.sidebar.selectbox(
        "Monitoring session",
        list(choices),
        key="heatmap_session",
        format_func=lambda key: (
            f'{local_time(choices[key]["started_at_utc"])}'
            f' | {choices[key]["source"]}'
            f' | {key[:8]}'
        ),
    )

    levels = st.sidebar.multiselect(
        "Highest incident level",
        LEVELS,
        default=LEVELS,
        key="heatmap_levels",
    )

    opacity = st.sidebar.slider(
        "Overlay strength",
        0.0,
        1.0,
        0.7,
        0.05,
    )

    smoothing = st.sidebar.slider(
        "Smoothing",
        0.0,
        3.0,
        1.0,
        0.5,
    )

    try:
        session, rows = read_heatmap(
            DATABASE_PATH,
            selected,
            levels,
        )

    except (
        OSError,
        sqlite3.Error,
        ValueError,
        KeyError,
    ) as error:
        st.error(
            f"Could not read this heatmap: {error}"
        )
        return

    total = sum(
        row["seconds"]
        for row in rows
    )

    incidents = len({
        row["incident_id"]
        for row in rows
    })

    cells = {}

    for row in rows:
        key = (
            row["cell_x"],
            row["cell_y"],
        )

        cells[key] = (
            cells.get(key, 0.0)
            + row["seconds"]
        )

    metrics = st.columns(3)

    metrics[0].metric(
        "Observed vehicle-seconds",
        f"{total:.2f}",
    )

    metrics[1].metric(
        "Linked incidents",
        incidents,
    )

    metrics[2].metric(
        "Peak cell vehicle-seconds",
        f"{max(cells.values(), default=0):.2f}",
    )

    st.caption(
        "Two vehicles observed for five seconds contribute "
        "ten vehicle-seconds. Missing observations and "
        "sampling gaps over two seconds add no time. "
        "Totals can therefore be lower than recorded "
        "incident durations."
    )

    st.caption(
        "Filters use each incident's highest recorded level. "
        "Colours are relative to this map: blue is lower "
        "occupancy, red is higher. Colour intensity is not "
        "a collision-risk score."
    )

    if not rows:
        st.info(
            "No spatial observations match these filters."
        )

    elif total == 0:
        st.info(
            "Observations exist, but more consecutive "
            "visible frames are needed to measure "
            "occupancy time."
        )

    try:
        image = render_heatmap(
            session,
            rows,
            opacity,
            smoothing,
        )

        st.image(
            image,
            channels="BGR",
            width="stretch",
        )

        success, encoded = cv2.imencode(
            ".png",
            image,
        )

        if success:
            st.download_button(
                "Download heatmap PNG",
                encoded.tobytes(),
                file_name=(
                    f"heatmap_{selected[:8]}.png"
                ),
                mime="image/png",
            )

        if st.checkbox(
            "Show original camera reference"
        ):
            st.image(
                session["reference_path"],
                width="stretch",
            )

    except (
        OSError,
        ValueError,
        cv2.error,
    ) as error:
        st.warning(
            "Could not display the saved camera "
            f"reference: {error}"
        )

    st.download_button(
        "Download spatial data CSV",
        cells_csv(rows),
        file_name=(
            f"heatmap_{selected[:8]}_cells.csv"
        ),
        mime="text/csv",
    )

    with st.expander(
        "Spatial data and session details"
    ):
        st.write({
            "Session": selected,
            "Source": session["source"],
            "Started": local_time(
                session["started_at_utc"]
            ),
            "Ended": (
                local_time(
                    session["ended_at_utc"]
                )
                if session["ended_at_utc"]
                else "Open / unfinished"
            ),
        })

        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
        )

    st.caption(
        "Each session has its own camera view. "
        "Restart and recalibrate if the camera moves."
    )