from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from auth import (
    require_auth,
    list_bank_accounts,
    render_admin_panel,
    logout,
)

from fraudlens_data import (
    init_db,
    get_profile,
    save_profile,
    suggest_mapping,
    create_upload,
    list_uploads,
    get_upload,
    load_upload_data,
    save_mapping_and_data,
    save_analysis,
    load_analysis,
    load_metrics,
    update_upload,
    delete_upload,
    upload_folder,
    list_all_institutions,
    list_all_institution_uploads,
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="FraudLens | Risk Management Platform",
    page_icon="🛡️",
    layout="wide",
)

init_db()

user = require_auth()

def render_admin_dashboard():
    """Super Admin view across all registered bank institutions."""

    st.title("FraudLens — Super Admin Dashboard")

    st.caption(
        "Central monitoring of registered institutions, uploaded datasets and transaction risk."
    )

    # Fetch all institution and upload records.
    institutions = list_all_institutions()
    uploads = list_all_institution_uploads()
    accounts = list_bank_accounts()

    # Convert SQLite records into dictionaries before DataFrame creation.
    institution_records = [dict(row) for row in institutions]
    upload_records = [dict(row) for row in uploads]
    account_records = [dict(row) for row in accounts]

    institution_df = pd.DataFrame(institution_records)
    upload_df = pd.DataFrame(upload_records)
    account_df = pd.DataFrame(account_records)

    # -------------------- Overview metrics --------------------
    total_institutions = len(accounts)
    active_accounts = 0

    if not account_df.empty and "is_active" in account_df.columns:
        active_accounts = int(
            account_df["is_active"].astype(int).sum()
        )

    total_uploads = len(uploads)
    total_records = 0

    if not upload_df.empty and "row_count" in upload_df.columns:
        total_records = int(
            pd.to_numeric(upload_df["row_count"], errors="coerce")
            .fillna(0).sum()
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("REGISTERED INSTITUTIONS", total_institutions)
    c2.metric("ACTIVE ACCOUNTS", active_accounts)
    c3.metric("TOTAL DATASET UPLOADS", total_uploads)
    c4.metric("TRANSACTION RECORDS", f"{total_records:,}")

    st.markdown("---")

    # -------------------- Institution monitoring --------------------
    st.markdown("### Registered Institutions")

    if account_df.empty:
        st.info("No bank institution accounts have been created yet.")
    else:
        # Show clear, human-readable account details.
        display_accounts = account_df.copy()

        if "is_active" in display_accounts.columns:
            display_accounts["Account Status"] = (
                pd.to_numeric(display_accounts["is_active"], errors="coerce")
                .fillna(0)
                .astype(int)
                .map({1: "Active", 0: "Deactivated"})
            )

        display_accounts = display_accounts.rename(columns={
            "id": "Institution ID",
            "institution_name": "Institution Name",
            "email": "Login Email",
            "created_at": "Created At",
        })

        preferred_columns = [
            column for column in [
                "Institution ID",
                "Institution Name",
                "Login Email",
                "Account Status",
                "Created At",
            ]
            if column in display_accounts.columns
        ]

        st.dataframe(
            display_accounts[preferred_columns],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # -------------------- Institution profiles --------------------
    st.markdown("### Institution Profiles")

    if institution_df.empty:
        st.info(
            "Institution profiles will appear here after bank users "
            "complete their Institution Profile."
        )
    else:
        st.dataframe(
            institution_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # -------------------- Dataset monitoring --------------------
    st.markdown("### All Uploaded Datasets")

    if upload_df.empty:
        st.info("No datasets have been uploaded by institutions yet.")
    else:
        visible_columns = [
            column for column in [
                "institution_name",
                "institution_user_id",
                "original_filename",
                "upload_time",
                "row_count",
                "column_count",
                "status",
                "status_message",
            ]
            if column in upload_df.columns
        ]

        st.dataframe(
            upload_df[visible_columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Dataset Activity Report",
            upload_df[visible_columns].to_csv(index=False).encode("utf-8"),
            file_name="fraudlens_dataset_activity.csv",
            mime="text/csv",
        )

    st.markdown("---")
        # -------------------- Combined risk analytics --------------------
    st.markdown("### Combined Transaction Risk Analytics")

    combined_results = []

    if not upload_df.empty:
        for _, upload in upload_df.iterrows():
            folder_path = upload.get("folder_path")

            if not folder_path:
                continue

            result_path = Path(folder_path) / "risk_results.csv"

            if result_path.exists():
                try:
                    result_df = pd.read_csv(result_path)

                    result_df["institution_name"] = upload.get(
                        "institution_name", "Unknown"
                    )
                    result_df["source_file"] = upload.get(
                        "original_filename", "Unknown"
                    )

                    combined_results.append(result_df)

                except Exception:
                    continue

    if combined_results:
        all_results = pd.concat(
            combined_results,
            ignore_index=True
        )

        total_transactions = len(all_results)

        flagged = (
            int((all_results["status"] == "FLAGGED").sum())
            if "status" in all_results.columns else 0
        )

        under_review = (
            int((all_results["status"] == "UNDER REVIEW").sum())
            if "status" in all_results.columns else 0
        )

        average_risk = (
            float(pd.to_numeric(
                all_results["risk_score"], errors="coerce"
            ).mean())
            if "risk_score" in all_results.columns else 0
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("ANALYZED TRANSACTIONS", f"{total_transactions:,}")
        r2.metric("FLAGGED TRANSACTIONS", f"{flagged:,}")
        r3.metric("UNDER REVIEW", f"{under_review:,}")
        r4.metric("AVERAGE RISK SCORE", f"{average_risk:.1f}%")

        left, right = st.columns(2)

        with left:
            if "status" in all_results.columns:
                risk_counts = (
                    all_results["status"]
                    .value_counts()
                    .rename_axis("Risk Status")
                    .reset_index(name="Transactions")
                )

                st.plotly_chart(
                    px.pie(
                        risk_counts,
                        names="Risk Status",
                        values="Transactions",
                        hole=0.55,
                        title="Combined Risk Distribution"
                    ),
                    use_container_width=True
                )

        with right:
            if (
                "institution_name" in all_results.columns
                and "status" in all_results.columns
            ):
                institution_risk = (
                    all_results.groupby(
                        ["institution_name", "status"]
                    )
                    .size()
                    .reset_index(name="Transactions")
                )

                st.plotly_chart(
                    px.bar(
                        institution_risk,
                        x="institution_name",
                        y="Transactions",
                        color="status",
                        barmode="group",
                        title="Institution-wise Risk Overview"
                    ),
                    use_container_width=True
                )

        st.markdown("### Highest-risk transactions across institutions")

        if "risk_score" in all_results.columns:
            all_results["risk_score"] = pd.to_numeric(
                all_results["risk_score"], errors="coerce"
            )

            st.dataframe(
                all_results.sort_values(
                    "risk_score", ascending=False
                ).head(25),
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "Download Combined Risk Report",
            all_results.to_csv(index=False).encode("utf-8"),
            file_name="fraudlens_combined_risk_report.csv",
            mime="text/csv",
        )

        st.caption(
            "Combined analytics include uploads that have completed risk analysis. "
            "Risk scores are prototype estimates, not proof of fraud."
        )

    else:
        st.info(
            "Combined risk analytics will appear after institutions upload "
            "datasets and complete their risk analysis."
        )


# Route the Super Admin to the central dashboard.

# ============================================================
# SUPER ADMIN NAVIGATION
# ============================================================

if user.get("role") == "admin":

    # Apply FraudLens dark navy and blue theme.
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(
                    ellipse at 78% 0%,
                    rgba(22,135,232,.13),
                    transparent 34%
                ),
                linear-gradient(
                    135deg,
                    #071426 0%,
                    #091a30 55%,
                    #071426 100%
                );
            color: #f1f5f9;
        }

        [data-testid="stHeader"] {
            background: rgba(7,20,38,.88);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg,#0b1b31,#0a192d);
            border-right: 1px solid #1e3652;
        }

        .block-container {
            max-width: 1550px;
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(145deg,#132b48,#0e2139);
            border: 1px solid #284563;
            border-radius: 15px;
            padding: 16px;
        }

        div[data-testid="stMetricLabel"] p {
            color: #a9bdd3 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fbff !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Administrator sidebar
    with st.sidebar:

        st.markdown("## 🛡️ FraudLens")

        st.caption("FINANCIAL SECURITY INTELLIGENCE")

        st.markdown("---")

        st.caption("SUPER ADMIN")

        st.write(user.get("email", ""))

        st.markdown("---")

        admin_page = st.radio(
            "ADMIN NAVIGATION",
            [
                "Super Admin Dashboard",
                "Administrator Console",
            ],
            key="super_admin_navigation",
        )

        st.markdown("---")

        if st.button(
            "🚪 Logout",
            key="super_admin_logout",
            use_container_width=True,
        ):
            logout()

    # Display selected page
    if admin_page == "Super Admin Dashboard":

        render_admin_dashboard()

    elif admin_page == "Administrator Console":

        render_admin_panel()

    st.stop()


# Continue with the existing bank institution application.
USER_ID = int(user["id"])

INSTITUTION_DIR = ROOT / "data" / "institutions" / str(USER_ID)
INSTITUTION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --fl-bg: #071426;
        --fl-panel: #10243d;
        --fl-border: #284563;
        --fl-blue: #38bdf8;
        --fl-muted: #9db0c8;
        --fl-text: #f1f5f9;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                ellipse at 78% 0%,
                rgba(22,135,232,.13),
                transparent 34%
            ),
            linear-gradient(
                135deg,
                #071426 0%,
                #091a30 55%,
                #071426 100%
            );
        color: var(--fl-text);
    }

    [data-testid="stHeader"] {
        background: rgba(7,20,38,.88);
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0b1b31,#0a192d);
        border-right: 1px solid #1e3652;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #c4d2e3;
    }

    h1, h2, h3, h4, p, label {
        color: inherit;
    }

    .panel {
        background: linear-gradient(145deg,#132b48,#0e2139);
        border: 1px solid #284563;
        border-radius: 16px;
        padding: 18px;
    }

    .main-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fbff;
        margin-bottom: .2rem;
    }

    .sub-title {
        color: #9db0c8;
        font-size: .92rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg,#132b48,#0e2139);
        border: 1px solid #284563;
        border-radius: 15px;
        padding: 16px;
        min-height: 105px;
    }

    div[data-testid="stMetricLabel"] p {
        color: #a9bdd3 !important;
        font-size: .72rem;
        font-weight: 750;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fbff !important;
        font-weight: 800;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 9px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

px.defaults.template = "plotly_dark"


# ============================================================
# COMMON HELPERS
# ============================================================

def title(main: str, subtitle: str = ""):
    st.markdown(
        f'<div class="main-title">{main}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f'<div class="sub-title">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def profile_complete(profile):
    required = [
        "institution_name",
        "institution_type",
        "country",
        "city",
        "state",
        "admin_name",
        "admin_email",
    ]

    return bool(
        profile
        and all(str(profile.get(field) or "").strip() for field in required)
    )


def get_active_upload_id(records):
    ids = {record["upload_id"] for record in records}

    selected = st.session_state.get("fraudlens_active_upload")

    if selected not in ids:
        selected = records[0]["upload_id"] if records else None
        st.session_state["fraudlens_active_upload"] = selected

    return selected


def get_thresholds(analysis):
    return analysis.get(
        "thresholds",
        {"high": 75, "medium": 45},
    )


# ============================================================
# RISK ANALYSIS ENGINE
# ============================================================

def make_risk_results(
    df: pd.DataFrame,
    positive_label: str | None = None,
):
    """
    Supervised classification is used only when a usable binary
    fraud label is supplied.

    Otherwise, Isolation Forest or a transparent fallback
    heuristic is used.
    """

    work = df.copy()
    metrics = pd.DataFrame()

    analysis = {
        "method": "Unsupervised anomaly detection",
        "labelled": False,
        "note": (
            "Anomaly scores indicate unusual patterns. "
            "They are not confirmed fraud probabilities."
        ),
    }

    label_col = (
        "fraud_label"
        if "fraud_label" in work.columns
        else None
    )

    leakage_columns = {
        "fraud_probability",
        "predicted_fraud",
        "predicted_status",
        "risk_score",
        "status",
        "prediction",
        "prediction_probability",
        "predicted_label",
        "fraud_prediction",
    }

    identifier_columns = {
        "transaction_id",
        "customer_id",
        "email",
        "phone",
        "device_id",
    }

    # --------------------------------------------------------
    # SUPERVISED RANDOM FOREST
    # --------------------------------------------------------

    if (
        label_col
        and positive_label is not None
        and work[label_col].notna().sum() >= 10
        and work[label_col].nunique(dropna=True) == 2
    ):
        labelled_mask = work[label_col].notna()

        labelled_data = work.loc[labelled_mask].copy()

        y = (
            labelled_data[label_col].astype(str)
            == str(positive_label)
        ).astype(int)

        feature_df = labelled_data.drop(
            columns=[label_col]
        ).copy()

        feature_df = feature_df.drop(
            columns=[
                column
                for column in feature_df.columns
                if column.lower() in (
                    identifier_columns | leakage_columns
                )
            ],
            errors="ignore",
        )

        if (
            len(feature_df.columns) > 0
            and y.nunique() == 2
            and y.value_counts().min() >= 2
        ):
            numeric_columns = (
                feature_df.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
            )

            categorical_columns = [
                column
                for column in feature_df.columns
                if column not in numeric_columns
            ]

            transformers = []

            if numeric_columns:
                transformers.append(
                    (
                        "num",
                        Pipeline(
                            [
                                (
                                    "imputer",
                                    SimpleImputer(strategy="median"),
                                ),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        numeric_columns,
                    )
                )

            if categorical_columns:
                transformers.append(
                    (
                        "cat",
                        Pipeline(
                            [
                                (
                                    "imputer",
                                    SimpleImputer(
                                        strategy="most_frequent"
                                    ),
                                ),
                                (
                                    "onehot",
                                    OneHotEncoder(
                                        handle_unknown="ignore"
                                    ),
                                ),
                            ]
                        ),
                        categorical_columns,
                    )
                )

            if transformers:
                prep = ColumnTransformer(transformers)

                model = Pipeline(
                    [
                        ("prep", prep),
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=150,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ]
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    feature_df,
                    y,
                    test_size=0.25,
                    random_state=42,
                    stratify=y,
                )

                model.fit(X_train, y_train)

                probabilities = model.predict_proba(X_test)[:, 1]
                predictions = (probabilities >= 0.5).astype(int)

                metric_values = {
                    "model": "Random Forest",
                    "accuracy": accuracy_score(
                        y_test, predictions
                    ),
                    "precision": precision_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    ),
                    "f1": f1_score(
                        y_test,
                        predictions,
                        zero_division=0,
                    ),
                }

                if len(np.unique(y_test)) == 2:
                    metric_values["roc_auc"] = roc_auc_score(
                        y_test,
                        probabilities,
                    )

                metrics = pd.DataFrame([metric_values])

                # Refit using all labelled records for row-level scores.
                model.fit(feature_df, y)

                all_probabilities = model.predict_proba(
                    feature_df
                )[:, 1]

                # Keep unlabelled records as NOT SCORED.
                work["risk_score"] = np.nan
                work["status"] = "NOT SCORED"

                work.loc[
                    labelled_mask, "risk_score"
                ] = np.round(all_probabilities * 100, 2)

                work.loc[
                    labelled_mask, "status"
                ] = np.where(
                    work.loc[labelled_mask, "risk_score"] >= 75,
                    "FLAGGED",
                    np.where(
                        work.loc[labelled_mask, "risk_score"] >= 45,
                        "UNDER REVIEW",
                        "APPROVED",
                    ),
                )

                analysis = {
                    "method": "Supervised Random Forest",
                    "labelled": True,
                    "positive_label": str(positive_label),
                    "thresholds": {
                        "high": 75,
                        "medium": 45,
                    },
                    "note": (
                        "Metrics are hold-out estimates on this "
                        "uploaded dataset. They are not proof of "
                        "real-world fraud detection performance."
                    ),
                }

                return work, metrics, analysis, model

    # --------------------------------------------------------
    # UNSUPERVISED NUMERIC FEATURES
    # --------------------------------------------------------

    numeric = work.select_dtypes(
        include=[np.number]
    ).copy()

    excluded = [
        column
        for column in numeric.columns
        if column.lower() in (
            {"fraud_label", "is_fraud", "label", "target"}
            | leakage_columns
            | identifier_columns
        )
    ]

    numeric = numeric.drop(
        columns=excluded,
        errors="ignore",
    )

    model = None

    if len(numeric.columns) and len(work) >= 5:
        matrix = numeric.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        matrix = matrix.fillna(
            matrix.median()
        ).fillna(0)

        detector = IsolationForest(
            contamination="auto",
            random_state=42,
        )

        detector.fit(matrix)

        raw_scores = -detector.decision_function(matrix)

        # Percentile ranking gives relative anomaly scores.
        scores = (
            pd.Series(raw_scores)
            .rank(method="average", pct=True)
            .to_numpy()
            * 100
        )

        work["risk_score"] = np.round(scores, 2)

        analysis["method"] = (
            "Isolation Forest anomaly detection"
        )

        analysis["thresholds"] = {
            "high": 90,
            "medium": 70,
        }

        model = detector

    # --------------------------------------------------------
    # AMOUNT-BASED FALLBACK
    # --------------------------------------------------------

    elif "amount" in work.columns:
        amounts = pd.to_numeric(
            work["amount"],
            errors="coerce",
        ).fillna(0)

        q = amounts.quantile(0.95)

        work["risk_score"] = np.where(
            q > 0,
            np.minimum(100, amounts / q * 70),
            0,
        )

        analysis["method"] = (
            "Amount-based demonstration heuristic"
        )

        analysis["thresholds"] = {
            "high": 80,
            "medium": 50,
        }

        analysis["note"] = (
            "This fallback ranks unusually large amounts. "
            "It is not a trained fraud classifier."
        )

    # --------------------------------------------------------
    # NO USABLE FEATURES
    # --------------------------------------------------------

    else:
        work["risk_score"] = 0.0

        analysis["method"] = (
            "No scoring method available"
        )

        analysis["thresholds"] = {
            "high": 100,
            "medium": 100,
        }

        analysis["note"] = (
            "No usable numeric transaction features were found. "
            "Review the column mapping and dataset."
        )

    # --------------------------------------------------------
    # METHOD-SPECIFIC STATUS THRESHOLDS
    # --------------------------------------------------------

    method = analysis.get("method", "")

    if method == "Isolation Forest anomaly detection":
        high_threshold = 90
        medium_threshold = 70

    elif method == "Amount-based demonstration heuristic":
        high_threshold = 80
        medium_threshold = 50

    else:
        high_threshold = 75
        medium_threshold = 45

    analysis["thresholds"] = {
        "high": high_threshold,
        "medium": medium_threshold,
    }

    work["status"] = np.where(
        work["risk_score"] >= high_threshold,
        "FLAGGED",
        np.where(
            work["risk_score"] >= medium_threshold,
            "UNDER REVIEW",
            "APPROVED",
        ),
    )

    return work, metrics, analysis, model


# ============================================================
# UPLOAD ANALYSIS AND RESULT STORAGE
# ============================================================

def run_upload_analysis(
    upload_id: str,
    positive_label: str | None = None,
):
    df = load_upload_data(USER_ID, upload_id)

    if df.empty:
        raise ValueError(
            "The standardized dataset is missing or empty. "
            "Confirm column mapping first."
        )

    results, metrics, details, model = make_risk_results(
        df,
        positive_label,
    )

    folder = upload_folder(USER_ID, upload_id)
    folder.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        folder / "risk_results.csv",
        index=False,
    )

    if model is not None:
        try:
            joblib.dump(
                model,
                folder / "model.joblib",
            )
        except Exception:
            pass

    details["records_analyzed"] = int(len(results))

    details["flagged_count"] = int(
        (results["status"] == "FLAGGED").sum()
    )

    details["review_count"] = int(
        (results["status"] == "UNDER REVIEW").sum()
    )

    details["average_risk"] = float(
        results["risk_score"].mean()
    ) if results["risk_score"].notna().any() else 0.0

    save_analysis(
        USER_ID,
        upload_id,
        details,
        metrics,
    )

    update_upload(
        USER_ID,
        upload_id,
        "Analysis complete",
        details["method"],
    )

    return results, metrics, details


def load_risk_results(upload_id: str):
    record = get_upload(USER_ID, upload_id)

    if not record:
        return pd.DataFrame()

    path = Path(record["folder_path"]) / "risk_results.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


# ============================================================
# PROFILE AND NAVIGATION
# ============================================================

profile = get_profile(USER_ID)
profile_ok = profile_complete(profile)

records = list_uploads(USER_ID)
active_upload_id = get_active_upload_id(records)


with st.sidebar:
    st.markdown("## 🛡️ FraudLens")
    st.caption("Financial Risk Intelligence Platform")
    st.markdown("---")

    if profile:
        st.caption(
            profile.get("display_name")
            or profile.get("institution_name")
        )

    pages = [
        "Institution Profile",
        "Dataset Upload",
        "Upload History",
        "Dashboard",
        "Risk Analysis",
        "Model Insights",
    ]

    if not profile_ok:
        page = "Institution Profile"
        st.warning(
            "Complete your institution profile to continue."
        )

    else:
        nav_version = st.session_state.get(
            "fraudlens_nav_version",
            0,
        )

        nav_key = f"fraudlens_nav_{nav_version}"

        requested_page = st.session_state.get(
            "fraudlens_page",
            "Dashboard",
        )

        if requested_page not in pages:
            requested_page = "Dashboard"

        page = st.radio(
            "NAVIGATION",
            pages,
            index=pages.index(requested_page),
            key=nav_key,
        )

        st.session_state["fraudlens_page"] = page

    st.markdown("---")
    st.markdown("**ACTIVE DATASET**")

    if records:
        options = {
            (
                f"{record['original_filename']} · "
                f"{record['upload_time'][:16]} · "
                f"{record['upload_id']}"
            ): record["upload_id"]
            for record in records
        }

        labels = list(options.keys())

        active_label = next(
            (
                key for key, value in options.items()
                if value == active_upload_id
            ),
            labels[0],
        )

        chosen_label = st.selectbox(
            "Select dataset",
            labels,
            index=labels.index(active_label),
            key=(
                f"fraudlens_active_dataset_selector_"
                f"{active_upload_id}"
            ),
        )

        if options[chosen_label] != active_upload_id:
            st.session_state["fraudlens_active_upload"] = (
                options[chosen_label]
            )
            st.rerun()

    else:
        st.caption("No datasets uploaded yet")

    st.markdown("---")
    st.caption(f"Signed in: {user.get('email', '')}")

    if st.button(
        "🚪 Logout",
        key="bank_logout",
        use_container_width=True,
    ):
        logout()


if page != "Institution Profile" and not profile_ok:
    st.warning(
        "Please complete your Institution Profile first."
    )
    st.stop()


# ============================================================
# INSTITUTION PROFILE
# ============================================================

if page == "Institution Profile":

    title(
        "Institution Profile",
        "Complete your institution information before uploading transaction data.",
    )

    defaults = profile or {}

    institution_types = [
        "Commercial Bank",
        "Cooperative Bank",
        "Small Finance Bank",
        "Payments Bank",
        "NBFC",
        "Fintech",
        "Other",
    ]

    registration_statuses = [
        "Pending verification",
        "Regulated",
        "Not regulated",
    ]

    currencies = [
        "INR",
        "USD",
        "EUR",
        "GBP",
        "Other",
    ]

    with st.form("institution_profile_form"):

        st.markdown("### Institution identity")

        c1, c2 = st.columns(2)

        institution_name = c1.text_input(
            "Official institution name *",
            value=(
                defaults.get("institution_name")
                or user.get("institution_name")
                or ""
            ),
        )

        display_name = c2.text_input(
            "Display name",
            value=defaults.get("display_name") or "",
        )

        logo_file = st.file_uploader(
            "Institution logo (optional)",
            type=["png", "jpg", "jpeg"],
            help="Upload your institution's logo.",
        )

        c3, c4 = st.columns(2)

        institution_type = c3.selectbox(
            "Institution type *",
            institution_types,
            index=(
                institution_types.index(
                    defaults["institution_type"]
                )
                if defaults.get("institution_type") in institution_types
                else 0
            ),
        )

        country = c4.text_input(
            "Country *",
            value=defaults.get("country") or "India",
        )

        c5, c6 = st.columns(2)

        established_year = c5.number_input(
            "Year established",
            min_value=1800,
            max_value=2100,
            value=int(
                defaults.get("established_year") or 2020
            ),
        )

        website = c6.text_input(
            "Official website",
            value=defaults.get("website") or "",
        )

        st.markdown(
            "### Registration and regulatory information"
        )

        c7, c8, c9 = st.columns(3)

        regulator = c7.text_input(
            "Regulator",
            value=defaults.get("regulator") or "",
        )

        registration_number = c8.text_input(
            "Registration / licence number",
            value=defaults.get("registration_number") or "",
        )

        registration_status = c9.selectbox(
            "Registration status",
            registration_statuses,
            index=(
                registration_statuses.index(
                    defaults.get("registration_status")
                )
                if defaults.get("registration_status")
                in registration_statuses
                else 0
            ),
        )

        st.caption(
            "Entering registration details does not independently "
            "verify a licence or regulatory status."
        )

        st.markdown("### Office and administrator")

        office_address = st.text_area(
            "Registered office address",
            value=defaults.get("office_address") or "",
        )

        c10, c11, c12 = st.columns(3)

        city = c10.text_input(
            "City *",
            value=defaults.get("city") or "",
        )

        state = c11.text_input(
            "State *",
            value=defaults.get("state") or "",
        )

        postal_code = c12.text_input(
            "Postal code",
            value=defaults.get("postal_code") or "",
        )

        c13, c14 = st.columns(2)

        admin_name = c13.text_input(
            "Administrator name *",
            value=defaults.get("admin_name") or "",
        )

        admin_email = c14.text_input(
            "Official administrator email *",
            value=(
                defaults.get("admin_email")
                or user.get("email")
                or ""
            ),
        )

        c15, c16 = st.columns(2)

        admin_designation = c15.text_input(
            "Designation / department",
            value=defaults.get("admin_designation") or "",
        )

        business_phone = c16.text_input(
            "Business phone",
            value=defaults.get("business_phone") or "",
        )

        fraud_contact_email = st.text_input(
            "Fraud / risk team contact email",
            value=defaults.get("fraud_contact_email") or "",
        )

        st.markdown("### Fraud monitoring configuration")

        channel_options = [
            "UPI",
            "Cards",
            "Online banking",
            "NEFT",
            "RTGS",
            "IMPS",
            "Wallets",
            "Other",
        ]

        try:
            selected_channels = json.loads(
                defaults.get("channels") or "[]"
            )
        except Exception:
            selected_channels = []

        channels = st.multiselect(
            "Transaction channels",
            channel_options,
            default=[
                value
                for value in selected_channels
                if value in channel_options
            ],
        )

        c17, c18 = st.columns(2)

        currency = c17.selectbox(
            "Primary currency",
            currencies,
            index=(
                currencies.index(defaults.get("currency"))
                if defaults.get("currency") in currencies
                else 0
            ),
        )

        daily_volume = c18.number_input(
            "Approximate daily transaction volume",
            min_value=0,
            value=int(
                defaults.get("approximate_daily_volume") or 0
            ),
            step=100,
        )

        submitted = st.form_submit_button(
            "Save Institution Profile",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        required = [
            institution_name.strip(),
            country.strip(),
            city.strip(),
            state.strip(),
            admin_name.strip(),
            admin_email.strip(),
        ]

        if not all(required):
            st.error(
                "Please complete all required fields marked with *."
            )

        elif "@" not in admin_email:
            st.error(
                "Please enter a valid administrator email address."
            )

        else:
            # Keep the previous logo if no new logo is uploaded.
            logo_path = defaults.get("logo_path")

            if logo_file is not None:
                logo_dir = INSTITUTION_DIR / "profile"
                logo_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                extension = Path(
                    logo_file.name
                ).suffix.lower()

                logo_path = logo_dir / (
                    f"institution_logo{extension}"
                )

                logo_path.write_bytes(
                    logo_file.getvalue()
                )

                logo_path = str(logo_path)

            save_profile(
                USER_ID,
                {
                    "institution_name": institution_name.strip(),
                    "display_name": display_name.strip(),
                    "institution_type": institution_type,
                    "country": country.strip(),
                    "regulator": regulator.strip(),
                    "registration_number": registration_number.strip(),
                    "registration_status": registration_status,
                    "established_year": int(established_year),
                    "website": website.strip(),
                    "office_address": office_address.strip(),
                    "city": city.strip(),
                    "state": state.strip(),
                    "postal_code": postal_code.strip(),
                    "admin_name": admin_name.strip(),
                    "admin_email": admin_email.strip(),
                    "admin_designation": admin_designation.strip(),
                    "business_phone": business_phone.strip(),
                    "fraud_contact_email": fraud_contact_email.strip(),
                    "channels": channels,
                    "currency": currency,
                    "approximate_daily_volume": int(daily_volume),
                    "logo_path": logo_path,
                },
            )

            st.success(
                "Institution profile saved successfully."
            )

            st.session_state["fraudlens_page"] = (
                "Dataset Upload"
            )

            st.session_state["fraudlens_nav_version"] = (
                st.session_state.get(
                    "fraudlens_nav_version",
                    0,
                ) + 1
            )

            st.rerun()

    # Display saved logo outside the form.
    saved_logo = defaults.get("logo_path")

    if saved_logo and Path(saved_logo).exists():
        st.markdown("### Saved institution logo")
        st.image(saved_logo, width=160)


# ============================================================
# DATASET UPLOAD
# ============================================================

elif page == "Dataset Upload":

    title(
        "Dataset Upload",
        "Upload CSV files with your institution's own column names. Previous uploads are preserved.",
    )

    uploader_version = st.session_state.get(
        "fraudlens_upload_widget_version",
        0,
    )

    uploaded_file = st.file_uploader(
        "Choose a transaction CSV file",
        type=["csv"],
        key=f"fraudlens_csv_upload_{uploader_version}",
        help=(
            "Choose a CSV file. Each confirmed upload "
            "is stored separately."
        ),
    )

    if uploaded_file is not None:

        try:
            file_bytes = uploaded_file.getvalue()

            file_signature = hashlib.sha256(
                file_bytes
            ).hexdigest()

            pending_id = st.session_state.get(
                "fraudlens_pending_upload"
            )

            pending_record = (
                get_upload(USER_ID, pending_id)
                if pending_id
                else None
            )

            if (
                pending_record
                and st.session_state.get(
                    "fraudlens_last_upload_signature"
                ) == file_signature
            ):
                new_id = pending_id

            elif (
                st.session_state.get(
                    "fraudlens_last_upload_signature"
                ) != file_signature
            ):

                raw = pd.read_csv(
                    BytesIO(file_bytes)
                )

                if raw.empty:
                    st.error(
                        "The uploaded CSV is empty. "
                        "Please choose a CSV containing transaction rows."
                    )
                    new_id = None

                elif len(raw.columns) < 1:
                    st.error(
                        "The CSV does not contain any columns."
                    )
                    new_id = None

                else:
                    new_id = create_upload(
                        USER_ID,
                        uploaded_file.name,
                        raw,
                    )

                    st.session_state[
                        "fraudlens_last_upload_signature"
                    ] = file_signature

                    st.session_state[
                        "fraudlens_last_upload_id"
                    ] = new_id

                    st.session_state[
                        "fraudlens_pending_upload"
                    ] = new_id

                    st.session_state[
                        "fraudlens_active_upload"
                    ] = new_id

                    st.success(
                        f"Upload saved as {new_id}. "
                        "Your original CSV has been preserved."
                    )

            else:
                new_id = None

                previous_id = st.session_state.get(
                    "fraudlens_last_upload_id"
                )

                if previous_id and get_upload(
                    USER_ID,
                    previous_id,
                ):
                    st.info(
                        "This file has already been uploaded "
                        "in this session. To upload it again as "
                        "a separate dataset, refresh the page "
                        "before selecting it again."
                    )

                else:
                    raw = pd.read_csv(
                        BytesIO(file_bytes)
                    )

                    if raw.empty:
                        st.error(
                            "The uploaded CSV is empty."
                        )
                    else:
                        new_id = create_upload(
                            USER_ID,
                            uploaded_file.name,
                            raw,
                        )

                        st.session_state[
                            "fraudlens_last_upload_signature"
                        ] = file_signature

                        st.session_state[
                            "fraudlens_last_upload_id"
                        ] = new_id

                        st.session_state[
                            "fraudlens_pending_upload"
                        ] = new_id

                        st.session_state[
                            "fraudlens_active_upload"
                        ] = new_id

            if new_id:
                st.session_state[
                    "fraudlens_pending_upload"
                ] = new_id

                st.session_state[
                    "fraudlens_active_upload"
                ] = new_id

                if st.session_state.get(
                    "fraudlens_upload_redirected"
                ) != new_id:

                    st.session_state[
                        "fraudlens_upload_redirected"
                    ] = new_id

                    st.rerun()

        except UnicodeDecodeError:
            st.error(
                "This CSV could not be decoded as text. "
                "Save it as UTF-8 CSV in Excel and upload it again."
            )

        except pd.errors.ParserError as exc:
            st.error(
                f"The CSV format could not be read: {exc}"
            )

        except Exception as exc:
            st.error(
                f"Unable to process this CSV: {exc}"
            )

    pending_id = st.session_state.get(
        "fraudlens_pending_upload"
    )

    pending = (
        get_upload(USER_ID, pending_id)
        if pending_id
        else None
    )

    if pending:

        raw = load_upload_data(
            USER_ID,
            pending_id,
            original=True,
        )

        if raw.empty:
            st.error(
                "The saved original CSV could not be found "
                "or contains no rows. Please upload the file again."
            )

        else:
            st.markdown("### Confirm detected columns")

            st.write(
                "FraudLens suggests standard fields based on "
                "column names. Review the suggestions and change "
                "any incorrect mapping before processing."
            )

            suggestions = suggest_mapping(raw)

            targets = [
                "(Ignore)",
                "transaction_id",
                "amount",
                "transaction_time",
                "hour",
                "merchant_name",
                "merchant_risk",
                "customer_id",
                "location",
                "distance_from_home_km",
                "device_change",
                "international",
                "transactions_last_24h",
                "account_age_days",
                "is_weekend",
                "fraud_label",
                "currency",
                "channel",
                "device_id",
            ]

            confirmed = {}

            with st.form(
                f"mapping_form_{pending_id}"
            ):

                st.caption(
                    "Review low-confidence suggestions. "
                    "Prediction-output columns should be set to Ignore."
                )

                for i, col in enumerate(raw.columns):

                    suggested = suggestions.get(
                        str(col),
                        {},
                    ).get(
                        "target",
                        "(Ignore)",
                    )

                    confidence = suggestions.get(
                        str(col),
                        {},
                    ).get(
                        "confidence",
                        0.0,
                    )

                    c1, c2, c3 = st.columns(
                        [2, 2, 1]
                    )

                    c1.write(f"**{col}**")

                    default_index = (
                        targets.index(suggested)
                        if suggested in targets
                        else 0
                    )

                    confirmed[str(col)] = c2.selectbox(
                        "Standard field",
                        targets,
                        index=default_index,
                        key=f"map_{pending_id}_{i}",
                        label_visibility="collapsed",
                    )

                    c3.caption(
                        f"{confidence:.0%} confidence"
                        if confidence
                        else "No match"
                    )

                confirm_mapping = st.form_submit_button(
                    "Confirm Mapping and Prepare Dataset",
                    type="primary",
                    use_container_width=True,
                )

            if confirm_mapping:

                selected = [
                    value
                    for value in confirmed.values()
                    if value != "(Ignore)"
                ]

                if len(selected) != len(set(selected)):
                    st.error(
                        "Two source columns cannot map to "
                        "the same standard field. Please correct "
                        "the mapping."
                    )

                else:
                    try:
                        standardized = save_mapping_and_data(
                            USER_ID,
                            pending_id,
                            confirmed,
                            raw,
                        )

                        st.session_state.pop(
                            "fraudlens_pending_upload",
                            None,
                        )

                        st.session_state.pop(
                            "fraudlens_last_upload_signature",
                            None,
                        )

                        st.session_state.pop(
                            "fraudlens_last_upload_id",
                            None,
                        )

                        st.session_state.pop(
                            "fraudlens_upload_redirected",
                            None,
                        )

                        st.session_state[
                            "fraudlens_active_upload"
                        ] = pending_id

                        st.session_state[
                            "fraudlens_upload_widget_version"
                        ] = uploader_version + 1

                        st.success(
                            f"Mapping confirmed. "
                            f"{len(standardized):,} records "
                            "are ready for analysis."
                        )

                        st.rerun()

                    except Exception as exc:
                        st.error(
                            f"Could not save the column mapping: {exc}"
                        )

            with st.expander(
                "Preview original uploaded data"
            ):
                st.dataframe(
                    raw.head(15),
                    use_container_width=True,
                    hide_index=True,
                )

    else:
        st.info(
            "Upload a CSV to begin. You can upload multiple "
            "datasets; each confirmed upload is stored independently."
        )


# ============================================================
# UPLOAD HISTORY
# ============================================================

elif page == "Upload History":

    title(
        "Upload History",
        "Select a previous upload to switch Dashboard, "
        "Risk Analysis and Model Insights to that dataset.",
    )

    records = list_uploads(USER_ID)

    if not records:
        st.info(
            "No uploads found yet. Upload your first CSV "
            "from Dataset Upload."
        )

    else:
        table = pd.DataFrame(
            [
                {
                    "Upload ID": record["upload_id"],
                    "File name": record["original_filename"],
                    "Uploaded": record["upload_time"],
                    "Records": record["row_count"],
                    "Columns": record["column_count"],
                    "Status": record["status"],
                }
                for record in records
            ]
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        labels = {
            (
                f"{record['original_filename']} | "
                f"{record['upload_id']} | "
                f"{record['status']}"
            ): record["upload_id"]
            for record in records
        }

        selected_label = st.selectbox(
            "Choose a historical dataset",
            list(labels.keys()),
            key="history_pick",
        )

        selected_id = labels[selected_label]

        c1, c2, c3 = st.columns(3)

        if c1.button(
            "Open selected dataset",
            type="primary",
            use_container_width=True,
        ):

            st.session_state[
                "fraudlens_active_upload"
            ] = selected_id

            st.session_state[
                "fraudlens_page"
            ] = "Dashboard"

            st.session_state[
                "fraudlens_nav_version"
            ] = (
                st.session_state.get(
                    "fraudlens_nav_version",
                    0,
                ) + 1
            )

            st.rerun()

        record = get_upload(
            USER_ID,
            selected_id,
        )

        original_path = (
            Path(record["folder_path"])
            / "original.csv"
        )

        if original_path.exists():
            c2.download_button(
                "Download original CSV",
                original_path.read_bytes(),
                file_name=record["original_filename"],
                mime="text/csv",
                use_container_width=True,
            )

        if c3.button(
            "Delete selected upload",
            use_container_width=True,
        ):
            st.session_state[
                "fraudlens_confirm_delete"
            ] = selected_id

        if st.session_state.get(
            "fraudlens_confirm_delete"
        ) == selected_id:

            st.warning(
                "Are you sure? This permanently deletes "
                "this upload and its saved analysis."
            )

            yes, no = st.columns(2)

            if yes.button(
                "Yes, delete",
                type="primary",
            ):

                delete_upload(
                    USER_ID,
                    selected_id,
                )

                st.session_state.pop(
                    "fraudlens_confirm_delete",
                    None,
                )

                if st.session_state.get(
                    "fraudlens_active_upload"
                ) == selected_id:

                    st.session_state[
                        "fraudlens_active_upload"
                    ] = None

                st.rerun()

            if no.button("Cancel"):
                st.session_state.pop(
                    "fraudlens_confirm_delete",
                    None,
                )
                st.rerun()

        original = load_upload_data(
            USER_ID,
            selected_id,
            original=True,
        )

        with st.expander(
            "Preview selected original dataset"
        ):
            st.dataframe(
                original.head(20),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# SELECTED DATASET CONTEXT
# ============================================================

else:

    records = list_uploads(USER_ID)

    active_upload_id = get_active_upload_id(
        records
    )

    if not active_upload_id:
        title(
            page,
            "Upload a dataset to see its analysis.",
        )

        st.info(
            "No dataset is selected. Go to Dataset Upload "
            "to add a CSV file."
        )

        st.stop()

    record = get_upload(
        USER_ID,
        active_upload_id,
    )

    if not record:
        st.warning(
            "The selected upload could not be found. "
            "Choose another upload from the sidebar."
        )
        st.stop()

    st.caption(
        f"Selected dataset: **{record['original_filename']}** "
        f"· {record['upload_id']} · {record['status']}"
    )

    if record["status"] == "Mapping required":
        st.warning(
            "This dataset is waiting for column mapping "
            "confirmation. Open Dataset Upload to complete "
            "the mapping."
        )
        st.stop()

    standardized = load_upload_data(
        USER_ID,
        active_upload_id,
    )

    if standardized.empty:
        st.warning(
            "The standardized dataset is not available. "
            "Please re-open Dataset Upload and confirm its mapping."
        )
        st.stop()

    analysis = load_analysis(
        USER_ID,
        active_upload_id,
    )

    risk_df = load_risk_results(
        active_upload_id
    )

    if analysis is None or risk_df.empty:

        title(
            "Analyze Selected Dataset",
            "Run fraud risk analysis for this upload. "
            "Results will be saved with this upload.",
        )

        label_col = (
            "fraud_label"
            if "fraud_label" in standardized.columns
            else None
        )

        positive_label = None

        if (
            label_col
            and standardized[label_col].dropna().nunique() == 2
        ):

            values = (
                standardized[label_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            positive_label = st.selectbox(
                "Select the value that represents FRAUD",
                values,
                key=f"positive_{active_upload_id}",
            )

        elif label_col:
            st.info(
                "A fraud-label column was mapped, but it does "
                "not contain exactly two non-empty classes. "
                "FraudLens will use anomaly detection instead."
            )

        else:
            st.info(
                "No fraud-label column was mapped. FraudLens "
                "will use anomaly detection where numeric "
                "features are available. No supervised accuracy "
                "metrics will be reported."
            )

        if st.button(
            "Run Analysis for This Dataset",
            type="primary",
        ):

            try:
                with st.spinner(
                    "Analyzing this dataset..."
                ):
                    run_upload_analysis(
                        active_upload_id,
                        positive_label,
                    )

                st.success(
                    "Analysis completed and saved for this upload."
                )

                st.rerun()

            except Exception as exc:
                st.error(
                    f"Analysis could not be completed: {exc}"
                )

        st.stop()

    thresholds = get_thresholds(analysis)

    high_threshold = int(
        thresholds.get("high", 75)
    )

    medium_threshold = int(
        thresholds.get("medium", 45)
    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "Dashboard":

        title(
            "Fraud Detection Dashboard",
            "Transaction overview and risk distribution "
            "for the selected upload.",
        )

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "TOTAL TRANSACTIONS",
            f"{len(risk_df):,}",
        )

        k2.metric(
            "FLAGGED TRANSACTIONS",
            f"{int((risk_df.status == 'FLAGGED').sum()):,}",
        )

        k3.metric(
            "UNDER REVIEW",
            f"{int((risk_df.status == 'UNDER REVIEW').sum()):,}",
        )

        average_risk = (
            risk_df["risk_score"].mean()
            if risk_df["risk_score"].notna().any()
            else 0
        )

        k4.metric(
            "AVERAGE RISK SCORE",
            f"{average_risk:.1f}%",
        )

        left, right = st.columns(2)

        with left:
            st.markdown(
                '<div class="panel">'
                '<b>Risk category distribution</b>'
                '</div>',
                unsafe_allow_html=True,
            )

            counts = (
                risk_df["status"]
                .value_counts()
                .rename_axis("Risk status")
                .reset_index(name="Transactions")
            )

            st.plotly_chart(
                px.pie(
                    counts,
                    names="Risk status",
                    values="Transactions",
                    hole=0.55,
                ),
                use_container_width=True,
            )

        with right:
            st.markdown(
                '<div class="panel">'
                '<b>Transaction amount distribution</b>'
                '</div>',
                unsafe_allow_html=True,
            )

            if "amount" in risk_df.columns:
                st.plotly_chart(
                    px.histogram(
                        risk_df,
                        x="amount",
                        color="status",
                        nbins=30,
                    ),
                    use_container_width=True,
                )
            else:
                st.info(
                    "No amount field is available in this dataset."
                )

        st.markdown(
            "### Highest-risk transactions"
        )

        st.dataframe(
            risk_df.sort_values(
                "risk_score",
                ascending=False,
                na_position="last",
            ).head(25),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            analysis.get("note", "")
        )


    # ========================================================
    # RISK ANALYSIS
    # ========================================================

    elif page == "Risk Analysis":

        title(
            "Risk Analysis",
            "Explore transaction-level risk scores "
            "for the selected upload.",
        )

        a, b = st.columns(2)

        with a:
            st.metric(
                "High risk",
                int(
                    (
                        risk_df["risk_score"]
                        >= high_threshold
                    ).sum()
                ),
            )

            st.metric(
                "Medium risk / review",
                int(
                    (
                        (risk_df["risk_score"] >= medium_threshold)
                        & (risk_df["risk_score"] < high_threshold)
                    ).sum()
                ),
            )

        with b:
            st.plotly_chart(
                px.histogram(
                    risk_df,
                    x="risk_score",
                    color="status",
                    nbins=25,
                    title="Risk score distribution",
                ),
                use_container_width=True,
            )

        if "amount" in risk_df.columns:
            hover_columns = [
                column
                for column in [
                    "transaction_id",
                    "merchant_name",
                    "location",
                ]
                if column in risk_df.columns
            ]

            st.plotly_chart(
                px.scatter(
                    risk_df,
                    x="amount",
                    y="risk_score",
                    color="status",
                    hover_data=hover_columns,
                    title="Transaction amount vs risk score",
                ),
                use_container_width=True,
            )

        st.markdown("### Filter transactions")

        statuses = st.multiselect(
            "Risk status",
            sorted(
                risk_df.status.dropna().unique().tolist()
            ),
            default=sorted(
                risk_df.status.dropna().unique().tolist()
            ),
        )

        min_score = st.slider(
            "Minimum risk score",
            0,
            100,
            0,
        )

        view = risk_df[
            risk_df.status.isin(statuses)
            & (risk_df.risk_score >= min_score)
        ]

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download risk analysis CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"{active_upload_id}_risk_analysis.csv"
            ),
            mime="text/csv",
        )

        st.warning(
            "Risk scores are educational prototype estimates. "
            "An anomaly or high score is not proof of fraud."
        )


    # ========================================================
    # MODEL INSIGHTS
    # ========================================================

    elif page == "Model Insights":

        title(
            "Model Insights",
            "Model information and evaluation metrics "
            "for the selected upload.",
        )

        st.info(
            f"Analysis method: **"
            f"{analysis.get('method', 'Not specified')}**"
        )

        st.write(
            analysis.get("note", "")
        )

        metrics = load_metrics(
            USER_ID,
            active_upload_id,
        )

        if not metrics.empty:

            st.markdown(
                "### Hold-out evaluation metrics"
            )

            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True,
            )

            metric_columns = [
                column
                for column in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                ]
                if column in metrics.columns
            ]

            if metric_columns:

                long = metrics.melt(
                    id_vars=(
                        ["model"]
                        if "model" in metrics.columns
                        else None
                    ),
                    value_vars=metric_columns,
                    var_name="Metric",
                    value_name="Score",
                )

                st.plotly_chart(
                    px.bar(
                        long,
                        x="Metric",
                        y="Score",
                        color=(
                            "model"
                            if "model" in long.columns
                            else None
                        ),
                        barmode="group",
                        range_y=[0, 1],
                    ),
                    use_container_width=True,
                )

        else:
            st.info(
                "No supervised model evaluation metrics are "
                "available for this upload. This is expected "
                "when the dataset has no usable fraud labels "
                "and anomaly detection is used."
            )

        st.caption(
            "Metrics, when shown, are hold-out estimates from "
            "this dataset and are not a guarantee of real-world "
            "performance."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "FraudLens | Financial Security Intelligence | "
    "Educational Prototype | Institution-specific datasets"
)