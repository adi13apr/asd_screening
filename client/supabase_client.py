import os
from supabase import create_client
from dotenv import load_dotenv

# Load .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        f"Supabase credentials not loaded. "
        f"SUPABASE_URL={SUPABASE_URL}, SUPABASE_KEY={'SET' if SUPABASE_KEY else None}"
    )

supabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)
