from __future__ import annotations

import json
import re
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "fraudlens_data.db"
UPLOAD_ROOT = ROOT / "data" / "institutions"


# ------------------------------------------------------------
# CANONICAL COLUMN NAMES AND SAFE ALIASES
# ------------------------------------------------------------

CANONICAL_FIELDS = {
    "transaction_id": [
        "transaction_id", "transactionid", "txn_id", "txn_ref",
        "reference", "reference_no", "transaction_ref", "id",
    ],
    "amount": [
        "amount", "transaction_amount", "txn_amount", "txn_value",
        "value", "amt", "payment_amount",
    ],
    "transaction_time": [
        "transaction_time", "timestamp", "txn_timestamp",
        "datetime", "date_time", "transaction_date", "txn_date", "date",
    ],
    "hour": ["hour", "transaction_hour", "txn_hour"],
    "merchant_name": [
        "merchant_name", "merchant", "payee", "beneficiary",
        "vendor", "store",
    ],
    "merchant_risk": [
        "merchant_risk", "merchant_risk_score", "risk_of_merchant",
    ],
    "customer_id": [
        "customer_id", "customerid", "cust_id", "cust_ref",
        "account_id", "user_id", "client_id",
    ],
    "location": [
        "location", "city", "merchant_location",
        "transaction_location", "country", "region",
    ],
    "distance_from_home_km": [
        "distance_from_home_km", "distance_from_home",
        "home_distance_km", "distance_km",
    ],
    "device_change": [
        "device_change", "new_device", "device_changed", "is_new_device",
    ],
    "international": [
        "international", "is_international", "cross_border",
    ],
    "transactions_last_24h": [
        "transactions_last_24h", "txn_count_24h",
        "transactions_24h", "transaction_count_24h",
    ],
    "account_age_days": [
        "account_age_days", "account_age", "customer_account_age_days",
    ],
    "is_weekend": ["is_weekend", "weekend", "weekend_flag"],
    "fraud_label": [
        "fraud_label", "is_fraud", "fraud", "label",
        "target", "class", "isfraud", "fraudulent",
    ],
    "currency": ["currency", "currency_code", "ccy"],
    "channel": [
        "channel", "transaction_channel", "payment_method", "method", "mode",
    ],
    "device_id": ["device_id", "device", "device_ref"],
}


PROFILE_FIELDS = [
    "institution_name",
    "display_name",
    "institution_type",
    "country",
    "regulator",
    "registration_number",
    "registration_status",
    "established_year",
    "website",
    "office_address",
    "city",
    "state",
    "postal_code",
    "admin_name",
    "admin_email",
    "admin_designation",
    "business_phone",
    "fraud_contact_email",
    "channels",
    "currency",
    "approximate_daily_volume",
    "branch_count",
    "branch_regions",
    "team_size",
    "alert_preferences",
    "logo_path",
]


# ------------------------------------------------------------
# DATABASE
# ------------------------------------------------------------

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _add_missing_columns(conn, table: str, definitions: dict[str, str]):
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }

    for column, definition in definitions.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )


def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS institution_profiles (
                institution_user_id INTEGER PRIMARY KEY,
                institution_name TEXT NOT NULL,
                display_name TEXT,
                institution_type TEXT NOT NULL,
                country TEXT NOT NULL,
                regulator TEXT,
                registration_number TEXT,
                registration_status TEXT,
                established_year INTEGER,
                website TEXT,
                office_address TEXT,
                city TEXT NOT NULL,
                state TEXT,
                postal_code TEXT,
                admin_name TEXT NOT NULL,
                admin_email TEXT NOT NULL,
                admin_designation TEXT,
                business_phone TEXT,
                fraud_contact_email TEXT,
                channels TEXT,
                currency TEXT DEFAULT 'INR',
                approximate_daily_volume INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Safe schema migration: existing profile data is retained.
        _add_missing_columns(
            conn,
            "institution_profiles",
            {
                "branch_count": "INTEGER",
                "branch_regions": "TEXT",
                "team_size": "INTEGER",
                "alert_preferences": "TEXT",
                "logo_path": "TEXT",
            },
        )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                upload_id TEXT PRIMARY KEY,
                institution_user_id INTEGER NOT NULL,
                original_filename TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                status_message TEXT,
                folder_path TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_uploads_user_time
            ON uploads(institution_user_id, upload_time)
        """)

        conn.commit()


# ------------------------------------------------------------
# INSTITUTION PROFILES
# ------------------------------------------------------------

def get_profile(user_id: int):
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM institution_profiles
            WHERE institution_user_id = ?
            """,
            (user_id,),
        ).fetchone()

    return dict(row) if row else None


def save_profile(user_id: int, values: dict[str, Any]):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    existing = get_profile(user_id)

    payload = {}

    for field in PROFILE_FIELDS:
        value = values.get(field)

        if field in {"channels", "branch_regions", "alert_preferences"}:
            if isinstance(value, str):
                payload[field] = value
            else:
                payload[field] = json.dumps(value or [])
        else:
            payload[field] = value

    with connect() as conn:
        if existing:
            assignments = ", ".join(
                f"{field} = ?" for field in PROFILE_FIELDS
            )

            conn.execute(
                f"""
                UPDATE institution_profiles
                SET {assignments}, updated_at = ?
                WHERE institution_user_id = ?
                """,
                [payload[field] for field in PROFILE_FIELDS]
                + [now, user_id],
            )
        else:
            columns = ["institution_user_id"] + PROFILE_FIELDS + [
                "created_at", "updated_at"
            ]

            placeholders = ", ".join(["?"] * len(columns))

            conn.execute(
                f"""
                INSERT INTO institution_profiles ({", ".join(columns)})
                VALUES ({placeholders})
                """,
                [user_id]
                + [payload[field] for field in PROFILE_FIELDS]
                + [now, now],
            )

        conn.commit()


# ------------------------------------------------------------
# COLUMN MAPPING
# ------------------------------------------------------------

def normalize_name(name: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(name).strip().lower(),
    ).strip("_")


def suggest_mapping(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Suggest mappings only for clear aliases.
    Avoids fuzzy matching that can silently map unrelated columns.
    """

    alias_lookup = {}

    for canonical, aliases in CANONICAL_FIELDS.items():
        for alias in aliases:
            alias_lookup[normalize_name(alias)] = canonical

    mapping = {}

    for column in df.columns:
        normalized = normalize_name(column)
        target = alias_lookup.get(normalized)

        confidence = 0.99 if target else 0.0

        mapping[str(column)] = {
            "target": target or "(Ignore)",
            "confidence": confidence,
        }

    # Prevent two source columns from receiving the same suggestion.
    used = set()

    for column in df.columns:
        target = mapping[str(column)]["target"]

        if target == "(Ignore)":
            continue

        if target in used:
            mapping[str(column)] = {
                "target": "(Ignore)",
                "confidence": 0.0,
            }
        else:
            used.add(target)

    return mapping


def apply_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    output = pd.DataFrame(index=df.index)

    # Canonical columns first.
    for source, target in mapping.items():
        if source not in df.columns:
            continue

        if target and target != "(Ignore)":
            if target in output.columns:
                raise ValueError(
                    f"Multiple source columns map to '{target}'."
                )

            output[target] = df[source]

    # Preserve ignored and unknown source columns using their
    # original names, unless that name is already represented.
    for source in df.columns:
        target = mapping.get(str(source), "(Ignore)")

        if target == "(Ignore)" and source not in output.columns:
            output[source] = df[source]

    # Standardize amount without deleting negative signs or decimals.
    if "amount" in output.columns:
        amount_text = (
            output["amount"]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.replace(r"[^0-9.\-]", "", regex=True)
        )
        output["amount"] = pd.to_numeric(
            amount_text,
            errors="coerce",
        )

    if "hour" in output.columns:
        output["hour"] = pd.to_numeric(
            output["hour"],
            errors="coerce",
        )

    if "transaction_time" in output.columns:
        parsed = pd.to_datetime(
            output["transaction_time"],
            errors="coerce",
        )
        output["transaction_time"] = parsed.astype("string")

    if "fraud_label" in output.columns:
        output["fraud_label"] = (
            output["fraud_label"].astype("string").str.strip()
        )

    return output


# ------------------------------------------------------------
# UPLOAD STORAGE
# ------------------------------------------------------------

def upload_folder(user_id: int, upload_id: str) -> Path:
    return (
        UPLOAD_ROOT
        / str(int(user_id))
        / "uploads"
        / str(upload_id)
    )


def create_upload(
    user_id: int,
    filename: str,
    raw: pd.DataFrame,
) -> str:
    if raw is None or raw.empty:
        raise ValueError("The uploaded CSV contains no transaction rows.")

    upload_id = "UP-" + uuid.uuid4().hex[:10].upper()
    folder = upload_folder(user_id, upload_id)
    folder.mkdir(parents=True, exist_ok=False)

    raw.to_csv(folder / "original.csv", index=False)

    now = datetime.now().astimezone().isoformat(timespec="seconds")

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO uploads (
                upload_id,
                institution_user_id,
                original_filename,
                upload_time,
                row_count,
                column_count,
                status,
                status_message,
                folder_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                upload_id,
                user_id,
                Path(filename).name,
                now,
                int(len(raw)),
                int(len(raw.columns)),
                "Mapping required",
                "Awaiting column mapping confirmation",
                str(folder),
            ),
        )
        conn.commit()

    return upload_id


def list_uploads(user_id: int):
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM uploads
            WHERE institution_user_id = ?
            ORDER BY upload_time DESC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def get_upload(user_id: int, upload_id: str):
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM uploads
            WHERE institution_user_id = ?
              AND upload_id = ?
            """,
            (user_id, upload_id),
        ).fetchone()

    return dict(row) if row else None


def update_upload(
    user_id: int,
    upload_id: str,
    status: str,
    message: str = "",
):
    with connect() as conn:
        conn.execute(
            """
            UPDATE uploads
            SET status = ?, status_message = ?
            WHERE institution_user_id = ? AND upload_id = ?
            """,
            (status, message, user_id, upload_id),
        )
        conn.commit()


def save_mapping_and_data(
    user_id: int,
    upload_id: str,
    mapping: dict[str, str],
    raw: pd.DataFrame,
):
    record = get_upload(user_id, upload_id)

    if not record:
        raise ValueError("Upload not found for this institution.")

    folder = Path(record["folder_path"]).resolve()
    expected = upload_folder(user_id, upload_id).resolve()

    if folder != expected:
        raise ValueError("Invalid upload storage path.")

    standardized = apply_mapping(raw, mapping)

    if standardized.empty:
        raise ValueError("The standardized dataset is empty.")

    (folder / "mapping.json").write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )

    standardized.to_csv(
        folder / "standardized.csv",
        index=False,
    )

    preprocessing = {
        "rows": int(len(standardized)),
        "columns": list(standardized.columns),
        "missing_values": int(
            standardized.isna().sum().sum()
        ),
        "duplicate_rows": int(
            standardized.duplicated().sum()
        ),
        "invalid_amounts": (
            int(standardized["amount"].isna().sum())
            if "amount" in standardized.columns
            else None
        ),
    }

    (folder / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2),
        encoding="utf-8",
    )

    update_upload(
        user_id,
        upload_id,
        "Ready for analysis",
        "Column mapping confirmed. Analysis can now be run.",
    )

    return standardized


def load_upload_data(
    user_id: int,
    upload_id: str,
    original: bool = False,
):
    record = get_upload(user_id, upload_id)

    if not record:
        return pd.DataFrame()

    folder = Path(record["folder_path"]).resolve()
    expected = upload_folder(user_id, upload_id).resolve()

    if folder != expected:
        raise ValueError("Invalid upload storage path.")

    filename = "original.csv" if original else "standardized.csv"
    path = folder / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path, low_memory=False)


# ------------------------------------------------------------
# ANALYSIS AND METRIC STORAGE
# ------------------------------------------------------------

def save_analysis(
    user_id: int,
    upload_id: str,
    result: dict[str, Any],
    metrics: pd.DataFrame | None = None,
):
    record = get_upload(user_id, upload_id)

    if not record:
        raise ValueError("Upload not found.")

    folder = Path(record["folder_path"]).resolve()
    expected = upload_folder(user_id, upload_id).resolve()

    if folder != expected:
        raise ValueError("Invalid upload storage path.")

    (folder / "analysis.json").write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )

    if metrics is not None:
        metrics.to_csv(
            folder / "model_metrics.csv",
            index=False,
        )


def load_analysis(user_id: int, upload_id: str):
    record = get_upload(user_id, upload_id)

    if not record:
        return None

    path = Path(record["folder_path"]) / "analysis.json"

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_metrics(user_id: int, upload_id: str):
    record = get_upload(user_id, upload_id)

    if not record:
        return pd.DataFrame()

    path = Path(record["folder_path"]) / "model_metrics.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError):
        return pd.DataFrame()


# ------------------------------------------------------------
# DELETE ONE INSTITUTION'S UPLOAD ONLY
# ------------------------------------------------------------

def delete_upload(user_id: int, upload_id: str):
    record = get_upload(user_id, upload_id)

    if not record:
        return False

    folder = Path(record["folder_path"]).resolve()
    institution_root = (
        UPLOAD_ROOT / str(int(user_id)) / "uploads"
    ).resolve()

    if institution_root not in folder.parents:
        raise ValueError("Invalid upload path. Deletion cancelled.")

    # Delete DB record first only after validating ownership/path.
    with connect() as conn:
        conn.execute(
            """
            DELETE FROM uploads
            WHERE institution_user_id = ? AND upload_id = ?
            """,
            (user_id, upload_id),
        )
        conn.commit()

    if folder.exists():
        shutil.rmtree(folder)

    return True

# ==========================================================
# ADMIN FUNCTIONS - ALL INSTITUTIONS AND DATASETS
# ==========================================================

def list_all_institutions():
    """
    Return institution profiles with their total upload counts.
    Used by the Super Admin dashboard.
    """
    init_db()

    with connect() as conn:
        rows = conn.execute("""
            SELECT
                p.institution_user_id,
                p.institution_name,
                p.display_name,
                p.country,
                p.city,
                p.admin_name,
                p.admin_email,
                p.created_at,
                p.updated_at,
                COUNT(u.upload_id) AS total_uploads
            FROM institution_profiles p
            LEFT JOIN uploads u
                ON p.institution_user_id = u.institution_user_id
            GROUP BY p.institution_user_id
            ORDER BY p.institution_name
        """).fetchall()

    return [dict(row) for row in rows]


def list_all_institution_uploads():
    """
    Return upload records belonging to every institution.
    Used by the Super Admin to monitor datasets.
    """
    init_db()

    with connect() as conn:
        rows = conn.execute("""
            SELECT
                u.upload_id,
                u.institution_user_id,
                COALESCE(
                    p.institution_name,
                    'Institution ' || u.institution_user_id
                ) AS institution_name,
                u.original_filename,
                u.upload_time,
                u.row_count,
                u.column_count,
                u.status,
                u.status_message,
                u.folder_path
            FROM uploads u
            LEFT JOIN institution_profiles p
                ON u.institution_user_id = p.institution_user_id
            ORDER BY u.upload_time DESC
        """).fetchall()

    return [dict(row) for row in rows]

def delete_institution_data(user_id: int):
    """
    Permanently delete one institution's profile, upload records,
    and stored dataset files.

    This function is called only after Admin confirms deletion.
    """
    import shutil

    user_id = int(user_id)

    # Identify this institution's dedicated folder.
    institution_folder = (
        UPLOAD_ROOT / str(user_id)
    ).resolve()

    upload_root = UPLOAD_ROOT.resolve()

    # Safety check: never allow deletion outside the institutions folder.
    if institution_folder.parent != upload_root:
        raise ValueError("Invalid institution folder path.")

    # Delete only this institution's files.
    if institution_folder.exists():
        shutil.rmtree(institution_folder)

    # Delete only this institution's database records.
    with connect() as conn:
        conn.execute(
            """
            DELETE FROM uploads
            WHERE institution_user_id = ?
            """,
            (user_id,)
        )

        conn.execute(
            """
            DELETE FROM institution_profiles
            WHERE institution_user_id = ?
            """,
            (user_id,)
        )

        conn.commit()

    return True
