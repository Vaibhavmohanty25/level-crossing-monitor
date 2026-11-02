"""Inspect Ollama's response without changing incident records."""

import json

from ollama import Client

from crossing_monitor.ollama_analysis import (
    DEFAULT_MODEL,
    analyse_incident,
    choose_incidents,
    ensure_table,
)
from crossing_monitor.storage import IncidentStore


class DiagnosticClient:
    def __init__(self, client):
        self.client = client

    def chat(self, **kwargs):
        response = self.client.chat(**kwargs)

        content = response.message.content or ""
        thinking = (
            getattr(response.message, "thinking", None) or ""
        )

        details = {
            "model": response.model,
            "done": response.done,
            "done_reason": response.done_reason,
            "generated_tokens": response.eval_count,
            "answer_characters": len(content.strip()),
            "thinking_characters": len(thinking),
        }

        print("\nRESPONSE DETAILS:")
        print(json.dumps(details, indent=2))

        # Return the original response to reproduce the parsing issue.
        return response


def main():
    store = IncidentStore()

    try:
        ensure_table(store)

        events = choose_incidents(
            store,
            limit=1,
            force=True,
        )

        if not events:
            print("No closed incident is available to check.")
            return

        client = Client(
            host="http://localhost:11434",
            timeout=600,
        )

        information = client.show(
            DEFAULT_MODEL
        ).model_dump()

        print("MODEL DETAILS:")
        print(
            json.dumps(
                {
                    "model": DEFAULT_MODEL,
                    "capabilities": information.get(
                        "capabilities"
                    ),
                    "thinking_controls": information.get(
                        "thinking"
                    ),
                },
                indent=2,
            )
        )

        print(
            "\nChecking incident:",
            events[0]["incident_id"][:8],
        )

        report = analyse_incident(
            DiagnosticClient(client),
            events[0],
            DEFAULT_MODEL,
        )

        print("\nAnalysis succeeded:")
        print(
            json.dumps(
                report["ai_visual_analysis"],
                indent=2,
            )
        )

    except Exception as error:
        print(
            f"\nDiagnostic result: "
            f"{type(error).__name__}: {error}"
        )

    finally:
        store.close()


if __name__ == "__main__":
    main()