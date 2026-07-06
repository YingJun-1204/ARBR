import argparse
import json
import sqlite3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--order", choices=["value", "lr_desc"], default="value")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            t.number,
            tv.value,
            t.state,
            tp.param_name,
            tp.param_value
        FROM trials t
        JOIN trial_values tv ON tv.trial_id = t.trial_id
        JOIN trial_params tp ON tp.trial_id = t.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY tv.value ASC, t.number ASC
        """
    ).fetchall()

    trials = {}
    for row in rows:
        item = trials.setdefault(
            row["number"],
            {"number": row["number"], "value": row["value"], "params": {}},
        )
        item["params"][row["param_name"]] = row["param_value"]

    if args.order == "lr_desc":
        ordered = sorted(
            trials.values(),
            key=lambda x: float(x["params"].get("learning_rate", 0.0)),
            reverse=True,
        )
    else:
        ordered = sorted(trials.values(), key=lambda x: x["value"])
    for trial in ordered[: args.limit]:
        p = trial["params"]
        print(
            json.dumps(
                {
                    "number": trial["number"],
                    "value": trial["value"],
                    "learning_rate": p.get("learning_rate"),
                    "num_gaussians": p.get("num_gaussians"),
                    "gate_lambda": p.get("gate_lambda"),
                    "gs_dropout": p.get("gs_dropout"),
                    "head_dropout": p.get("head_dropout"),
                    "gs_residual_weight": p.get("gs_residual_weight"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
