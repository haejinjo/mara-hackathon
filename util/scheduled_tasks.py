import logging
import os

import httpx
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the project root
load_dotenv()


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
            print(data)
            logger.info(f"Successfully retrieved data from {prices_url}: {data}")
            # You can now process the data, for example, save it to a database.
    except httpx.RequestError as exc:
        logger.error(f"An error occurred while requesting {exc.request.url!r}: {exc}")


async def another_daily_task():
    """
    Another example of a scheduled task.
    """
    logger.info("Executing another_daily_task...")
    # Add your logic for the other task here
    logger.info("Another daily task finished.")