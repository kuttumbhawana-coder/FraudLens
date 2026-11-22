import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path

import streamlit as st


# ==========================================================
# DATABASE CONFIGURATION
# ==========================================================

DB_PATH = Path(__file__).resolve().parent / "fraudlens_auth.db"
ITERATIONS = 310_000


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK(role IN ('admin', 'bank')),
                institution_name TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


# ==========================================================
# PASSWORD SECURITY
# ==========================================================

def hash_password(password):
    salt = secrets.token_bytes(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        ITERATIONS
    )

    return f"{salt.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
    try:
        salt_hex, digest_hex = stored_hash.split("$", 1)

        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            ITERATIONS
        ).hex()

        return hmac.compare_digest(candidate, digest_hex)

    except (ValueError, TypeError):
        return False


# ==========================================================
# SUPER ADMIN
# ==========================================================

def bootstrap_admin():
    """
    Create the first Super Admin using credentials stored
    in .streamlit/secrets.toml.
    """

    try:
        email = str(st.secrets["ADMIN_EMAIL"]).strip().lower()
        password = str(st.secrets["ADMIN_PASSWORD"])
    except Exception:
        return

    if not email or len(password) < 12:
        return

    with get_connection() as conn:

        existing_admin = conn.execute("""
            SELECT id
            FROM users
            WHERE role = 'admin'
            LIMIT 1
        """).fetchone()

        if existing_admin is None:

            conn.execute("""
                INSERT INTO users
                (email, password_hash, role, institution_name)
                VALUES (?, ?, 'admin', 'FraudLens Administration')
            """, (
                email,
                hash_password(password)
            ))

            conn.commit()


# ==========================================================
# AUTHENTICATION
# ==========================================================

def authenticate(email, password):

    with get_connection() as conn:

        user = conn.execute("""
            SELECT *
            FROM users
            WHERE email = ? COLLATE NOCASE
        """, (email.strip(),)).fetchone()

    if user is None:
        return None

    if not user["is_active"]:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return {
        "id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "institution_name": user["institution_name"]
    }


# ==========================================================
# CREATE BANK ACCOUNT
# ==========================================================

def create_bank_account(institution_name, email, password):

    institution_name = institution_name.strip()
    email = email.strip().lower()

    if not institution_name or not email or not password:
        raise ValueError("Please fill in all fields.")

    if len(password) < 12:
        raise ValueError(
            "Password must contain at least 12 characters."
        )

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO users
            (institution_name, email, password_hash, role, is_active)
            VALUES (?, ?, ?, 'bank', 1)
        """, (
            institution_name,
            email,
            hash_password(password)
        ))

        conn.commit()


# ==========================================================
# LIST BANK ACCOUNTS
# ==========================================================

def list_bank_accounts():

    with get_connection() as conn:

        return conn.execute("""
            SELECT id, institution_name, email,
                   is_active, created_at
            FROM users
            WHERE role = 'bank'
            ORDER BY created_at DESC
        """).fetchall()


# ==========================================================
# ACTIVATE / DEACTIVATE BANK
# ==========================================================

def set_bank_active(user_id, active):

    with get_connection() as conn:

        conn.execute("""
            UPDATE users
            SET is_active = ?
            WHERE id = ? AND role = 'bank'
        """, (
            1 if active else 0,
            int(user_id)
        ))

        conn.commit()


# ==========================================================
# DELETE BANK INSTITUTION AND ITS DATA
# ==========================================================

def delete_bank_account(user_id):
    """
    Permanently delete a bank institution and its associated
    datasets, upload records and institution profile.

    The Super Admin account cannot be deleted using this function.
    """

    user_id = int(user_id)

    # Verify that the selected account is a bank account.
    with get_connection() as conn:

        bank = conn.execute("""
            SELECT id
            FROM users
            WHERE id = ? AND role = 'bank'
        """, (user_id,)).fetchone()

    if bank is None:
        raise ValueError("Bank institution account not found.")

    # Delete the institution's associated data.
    from fraudlens_data import delete_institution_data

    delete_institution_data(user_id)

    # Delete the bank login account.
    with get_connection() as conn:

        conn.execute("""
            DELETE FROM users
            WHERE id = ? AND role = 'bank'
        """, (user_id,))

        conn.commit()

    return True


# ==========================================================
# COMMON DESIGN
# ==========================================================

def apply_dark_theme():

    st.html("""
    <style>

    .stApp {
        background-color: #0b1428;
        color: #ffffff;
    }

    [data-testid="stHeader"] {
        background-color: #0b1428;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #203d62;
        color: white;
        border: 1px solid #42658d;
        border-radius: 10px;
        min-height: 48px;
        font-weight: bold;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #2b5683;
        color: white;
        border-color: #62b5ff;
    }

    [data-testid="stSidebar"] {
        background-color: #101e35;
    }

    </style>
    """)


# ==========================================================
# LANDING PAGE
# ==========================================================

def render_landing_page():

    apply_dark_theme()

    st.html("""
    <style>

    .brand-header {
        display: flex;
        align-items: center;
        gap: 15px;
        border-bottom: 1px solid #34445e;
        padding-bottom: 25px;
        margin-bottom: 35px;
    }

    .brand-icon {
        background: #203d62;
        padding: 14px;
        border-radius: 18px;
        font-size: 30px;
    }

    .brand-name {
        color: white;
        font-size: 30px;
        font-weight: bold;
    }

    .brand-subtitle {
        color: #a9bbd3;
        font-size: 16px;
    }

    .secure-badge {
        margin-left: auto;
        color: #00d477;
        background: #10372f;
        padding: 8px 15px;
        border-radius: 25px;
        font-size: 14px;
    }

    .hero-title {
        color: white;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 12px;
    }

    .hero-title span {
        color: #62b5ff;
    }

    .hero-description {
        color: #b5c4d9;
        font-size: 19px;
        margin-bottom: 25px;
    }

    .portal-card {
        background: #1c304c;
        padding: 24px;
        border-radius: 20px;
        min-height: 185px;
        border: 1px solid #263e5e;
    }

    .portal-icon {
        color: #62b5ff;
        font-size: 30px;
    }

    .portal-card h3 {
        color: white;
        font-size: 23px;
    }

    .portal-card p {
        color: #b5c4d9;
        font-size: 17px;
    }

    .portal-card b {
        color: white;
        font-size: 17px;
    }

    .footer-text {
        color: #a9bbd3;
        font-size: 16px;
        border-top: 1px solid #34445e;
        padding-top: 25px;
        margin-top: 30px;
    }

    </style>

    <div class="brand-header">

        <div class="brand-icon">🛡️</div>

        <div>
            <div class="brand-name">FraudLens</div>
            <div class="brand-subtitle">
                FINANCIAL SECURITY INTELLIGENCE
            </div>
        </div>

        <div class="secure-badge">
            ● SECURE PORTAL
        </div>

    </div>

    <div class="hero-title">
        Protecting transactions.
        <span>Building trust.</span>
    </div>

    <div class="hero-description">
        Intelligent fraud detection and transaction risk
        analytics for financial institutions.
    </div>
    """)

    col1, col2 = st.columns(2, gap="medium")

    with col1:

        st.html("""
        <div class="portal-card">
            <div class="portal-icon">🛡️</div>
            <h3>Administrator</h3>
            <p>Manage institutions and platform access.</p>
            <b>Admin Portal →</b>
        </div>
        """)

        if st.button(
            "Enter Admin Portal",
            use_container_width=True,
            key="admin_portal_button"
        ):

            st.session_state["login_portal"] = "admin"
            st.rerun()

    with col2:

        st.html("""
        <div class="portal-card">
            <div class="portal-icon">🏦</div>
            <h3>Bank Institution</h3>
            <p>Monitor transactions and fraud risks.</p>
            <b>Institution Portal →</b>
        </div>
        """)

        if st.button(
            "Enter Institution Portal",
            use_container_width=True,
            key="bank_portal_button"
        ):

            st.session_state["login_portal"] = "bank"
            st.rerun()

    st.html("""
    <div class="footer-text">
        AI-POWERED RISK ANALYTICS
        <span style="float:right;">FRAUDLENS © 2026</span>
    </div>
    """)


# ==========================================================
# LOGIN PAGE
# ==========================================================

def render_login(portal):

    apply_dark_theme()

    if portal == "admin":

        title = "Administrator Login"
        description = "Authorized Super Admin access only."

    else:

        title = "Bank Institution Login"
        description = (
            "Login using your registered institution credentials."
        )

    st.html(f"""
    <div style="
        background:#1c304c;
        border:1px solid #34445e;
        border-radius:18px;
        padding:30px;
        text-align:center;
        margin:30px auto 20px auto;
        max-width:650px;
    ">

        <div style="font-size:40px;">🛡️</div>

        <h1 style="color:white;">FraudLens</h1>

        <p style="color:#a9bbd3;">
            FINANCIAL SECURITY INTELLIGENCE
        </p>

        <h3 style="color:#62b5ff;">{title}</h3>

        <p style="color:#b5c4d9;">{description}</p>

    </div>
    """)

    with st.form(key=f"{portal}_login_form"):

        email = st.text_input("Registered email address")

        password = st.text_input(
            "Password",
            type="password"
        )

        submitted = st.form_submit_button(
            "Login",
            use_container_width=True
        )

        if submitted:

            if not email.strip() or not password:

                st.error("Please enter email and password.")

            else:

                user = authenticate(email, password)

                if user is None:

                    st.error(
                        "Invalid credentials or inactive account."
                    )

                elif user["role"] != portal:

                    st.error(
                        "This account is not authorized for this portal."
                    )

                else:

                    st.session_state["fraudlens_user"] = user
                    st.session_state.pop("login_portal", None)
                    st.rerun()

    if st.button(
        "← Back to Portal Selection",
        use_container_width=True
    ):

        st.session_state.pop("login_portal", None)
        st.rerun()


# ==========================================================
# ADMIN CONSOLE
# ==========================================================

def render_admin_panel():

    apply_dark_theme()

    st.title("🛡️ Administrator Console")

    st.caption("Super Admin – Institution Management")

    st.markdown("---")

    # ------------------------------------------------------
    # CREATE BANK ACCOUNT
    # ------------------------------------------------------

    with st.expander(
        "➕ Register New Bank Institution",
        expanded=True
    ):

        with st.form(
            "create_bank_form",
            clear_on_submit=True
        ):

            institution = st.text_input(
                "Bank / Institution Name"
            )

            email = st.text_input(
                "Official Login Email"
            )

            password = st.text_input(
                "Temporary Password (minimum 12 characters)",
                type="password"
            )

            submitted = st.form_submit_button(
                "Create Bank Account"
            )

        if submitted:

            try:

                create_bank_account(
                    institution,
                    email,
                    password
                )

                st.success(
                    "Bank account created successfully!"
                )

                st.rerun()

            except sqlite3.IntegrityError:

                st.error(
                    "An account with this email already exists."
                )

            except ValueError as error:

                st.error(str(error))

    st.markdown("---")

    # ------------------------------------------------------
    # REGISTERED INSTITUTIONS
    # ------------------------------------------------------

    st.subheader("Registered Bank Institutions")

    banks = list_bank_accounts()

    if not banks:

        st.info("No bank accounts registered yet.")
        return

    st.write(f"Total institutions: **{len(banks)}**")

    for bank in banks:

        bank_id = bank["id"]

        with st.container(border=True):

            col1, col2, col3 = st.columns([3, 2, 2])

            with col1:

                st.write(
                    f"**{bank['institution_name']}**"
                )

                st.caption(bank["email"])
                st.caption(f"Account ID: {bank_id}")

            with col2:

                if bank["is_active"]:
                    st.success("Active")
                else:
                    st.warning("Deactivated")

                st.caption(
                    f"Created: {bank['created_at']}"
                )

            with col3:

                # ACTIVATE / DEACTIVATE
                if bank["is_active"]:

                    if st.button(
                        "Deactivate",
                        key=f"deactivate_{bank_id}",
                        use_container_width=True
                    ):

                        set_bank_active(bank_id, False)
                        st.rerun()

                else:

                    if st.button(
                        "Activate",
                        key=f"activate_{bank_id}",
                        use_container_width=True
                    ):

                        set_bank_active(bank_id, True)
                        st.rerun()

                # DELETE INSTITUTION
                if st.button(
                    "Delete Institution",
                    key=f"delete_{bank_id}",
                    use_container_width=True
                ):

                    st.session_state[
                        "confirm_delete_bank"
                    ] = bank_id

            # ------------------------------------------------
            # DELETE CONFIRMATION
            # ------------------------------------------------

            if st.session_state.get(
                "confirm_delete_bank"
            ) == bank_id:

                st.error(
                    f"Are you sure you want to permanently "
                    f"delete {bank['institution_name']} "
                    f"and its associated data?"
                )

                yes_col, no_col = st.columns(2)

                with yes_col:

                    if st.button(
                        "Yes, permanently delete",
                        key=f"confirm_yes_{bank_id}",
                        type="primary",
                        use_container_width=True
                    ):

                        try:

                            delete_bank_account(bank_id)

                            st.session_state.pop(
                                "confirm_delete_bank",
                                None
                            )

                            st.success(
                                "Institution deleted successfully."
                            )

                            st.rerun()

                        except Exception as error:

                            st.error(
                                f"Deletion failed: {error}"
                            )

                with no_col:

                    if st.button(
                        "Cancel",
                        key=f"confirm_no_{bank_id}",
                        use_container_width=True
                    ):

                        st.session_state.pop(
                            "confirm_delete_bank",
                            None
                        )

                        st.rerun()


# ==========================================================
# LOGOUT
# ==========================================================

def logout():

    st.session_state.pop("fraudlens_user", None)
    st.session_state.pop("login_portal", None)

    st.rerun()


# ==========================================================
# MAIN AUTHENTICATION CONTROLLER
# ==========================================================

def require_auth():

    initialize_database()
    bootstrap_admin()

    user = st.session_state.get("fraudlens_user")

    # ------------------------------------------------------
    # SHOW LANDING PAGE OR LOGIN PAGE
    # ------------------------------------------------------

    if not user:

        selected_portal = st.session_state.get("login_portal")

        if selected_portal in ("admin", "bank"):

            render_login(selected_portal)

        else:

            render_landing_page()

        st.stop()

    # ------------------------------------------------------
    # VERIFY ACCOUNT STATUS
    # ------------------------------------------------------

    with get_connection() as conn:

        db_user = conn.execute("""
            SELECT id, email, role,
                   institution_name, is_active
            FROM users
            WHERE id = ?
        """, (user["id"],)).fetchone()

    if (
        db_user is None
        or not db_user["is_active"]
        or db_user["role"] != user["role"]
    ):

        logout()

    user = {
        "id": db_user["id"],
        "email": db_user["email"],
        "role": db_user["role"],
        "institution_name": db_user["institution_name"]
    }

    st.session_state["fraudlens_user"] = user

    return user