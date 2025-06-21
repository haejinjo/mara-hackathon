import logging
import os
import asyncio
import sqlite3

import httpx
from dotenv import load_dotenv

from db.db import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the project root
load_dotenv()


def _insert_pricing_data_sync(data_to_insert: list[dict]):
    """
    Synchronous function to insert a list of pricing data into the database.
    It handles unique constraint violations by logging a warning and skipping.
    """
    # Using a single connection for all inserts in this batch
    conn = get_db_connection()
    try:
        with conn:  # `with` statement handles transactions (commit/rollback)
            for item in data_to_insert:
                try:
                    conn.execute(
                        """
                        INSERT INTO pricing_data (energy_price, hash_price, timestamp, token_price)
                        VALUES (:energy_price, :hash_price, :timestamp, :token_price)
                        """,
                        item,
                    )
                except sqlite3.IntegrityError:
                    # This is expected if the timestamp is already in the DB due to the UNIQUE constraint.
                    logger.warning(f"Timestamp '{item.get('timestamp')}' already exists. Skipping record.")
        logger.info(f"Successfully processed {len(data_to_insert)} records for the database.")
    except sqlite3.Error as e:
        logger.error(f"A database error occurred: {e}")
    finally:
        if conn:
            conn.close()


async def my_scheduled_task():
    """
    A scheduled task that makes a request to the MARA_ENDPOINT/prices.
    """
    logger.info("Executing my_scheduled_task...")
    mara_endpoint = os.getenv("MARA_ENDPOINT")
    if not mara_endpoint:
        logger.error("MARA_ENDPOINT environment variable not set.")
        return

    prices_url = f"{mara_endpoint}/prices"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(prices_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            logger.info(f"Successfully retrieved data from {prices_url}")

            if isinstance(data, list) and data:
                # Run the synchronous database operation in a separate thread
                # to avoid blocking the asyncio event loop.
                await asyncio.to_thread(_insert_pricing_data_sync, data)
            elif not data:
                logger.info("No new pricing data received from the endpoint.")
            else:
                logger.warning(f"Data received from endpoint is not a list: {type(data)}")
    except httpx.RequestError as exc:
        logger.error(f"An error occurred while requesting {exc.request.url!r}: {exc}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in my_scheduled_task: {e}", exc_info=True)

async def another_daily_task():
    """
    Another example of a scheduled task.
    """
    logger.info("Executing another_daily_task...")
    # Add your logic for the other task here
    logger.info("Another daily task finished.")