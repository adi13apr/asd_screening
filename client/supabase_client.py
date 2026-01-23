import os
from supabase import create_client
from dotenv import load_dotenv

# Load .env for local development only; production will provide env vars
load_dotenv()

def get_supabase_client():
    """Create and return a Supabase client using environment variables.

    This avoids creating the client at import time (which can fail on serverless platforms
    if secrets are not yet provided). Call this during your app startup.
    """
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            f"Supabase credentials not loaded. SUPABASE_URL={SUPABASE_URL}, SUPABASE_KEY={'SET' if SUPABASE_KEY else None}"
        )

    return create_client(SUPABASE_URL, SUPABASE_KEY)

