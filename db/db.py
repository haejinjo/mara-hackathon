# db/database_raw.py
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DATABASE_FILE = "mara_hackathon.db" # Name of your SQLite database file

def get_db_connection():
    """
    Establishes and returns a new SQLite database connection.
    Configures row_factory to return rows as dictionaries for easier access.
    """
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row # This makes rows behave like dicts (e.g., row['column_name'])
    return conn

def create_tables():
    """
    Creates the necessary tables if they don't already exist.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pricing_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            energy_price REAL NOT NULL,
            hash_price REAL NOT NULL,
            timestamp TEXT NOT NULL UNIQUE, -- Storing as TEXT (ISO 8601 format) and making it unique
            token_price REAL NOT NULL
        );
        """)
        conn.commit()
        logger.info(f"Database table 'pricing_data' ensured in {DATABASE_FILE}.")
    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")
    finally:
        if conn:
            conn.close()

# Ensure tables are created when this module is imported (optional, but convenient for first run)
# It's better to call this explicitly in the lifespan event of FastAPI.
# create_tables()