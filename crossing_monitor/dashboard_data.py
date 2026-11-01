"""Read incident history without modifying the monitoring database."""

import csv
import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def connect_readonly(path):
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Incident database not found: {path}"
        )

    connection = sqlite3.connect(
        path.as_uri() + "?mode=ro",
        uri=True,
        timeout=5,
    )

    connection.row_factory = sqlite3.Row

    return connection


def read_incidents(path, limit=1000):
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError(
            "Limit must be a positive integer."
        )

    with closing(connect_readonly(path)) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM incidents
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def read_details(path, incident_id):
    with closing(connect_readonly(path)) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM incidents
            WHERE incident_id = ?
            """,
            (incident_id,),
        ).fetchone()

        if row is None:
            raise KeyError(
                "The selected incident no longer exists."
            )

        event = dict(row)

        snapshot_rows = connection.execute(
            """
            SELECT *
            FROM snapshots
            WHERE incident_id = ?
            ORDER BY snapshot_id
            """,
            (incident_id,),
        ).fetchall()

        event["snapshots"] = [
            dict(item) for item in snapshot_rows
        ]

        report = None

        table_exists = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE
                type = 'table'
                AND name = 'incident_analyses'
            """
        ).fetchone()

        if table_exists:
            analysis = connection.execute(
                """
                SELECT report_json
                FROM incident_analyses
                WHERE incident_id = ?
                """,
                (incident_id,),
            ).fetchone()

            if analysis:
                report = json.loads(
                    analysis["report_json"]
                )

                if not isinstance(report, dict):
                    raise ValueError(
                        "Saved report must be a JSON object."
                    )

    return event, report


def filter_incidents(
    records,
    levels,
    status=None,
    source=None,
    minimum_duration=0,
):
    return [
        row
        for row in records
        if row["max_level"] in levels
        and (
            status is None
            or row["status"] == status
        )
        and (
            source is None
            or row["source"] == source
        )
        and row["duration_seconds"] >= minimum_duration
    ]


def records_to_csv(records):
    if not records:
        return ""

    output = io.StringIO(newline="")

    writer = csv.DictWriter(
        output,
        fieldnames=list(records[0]),
    )

    writer.writeheader()
    writer.writerows(records)

    return output.getvalue()