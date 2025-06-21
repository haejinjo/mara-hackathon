# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

# Import your scheduled task functions
from util.scheduled_tasks import my_scheduled_task, another_daily_task

# Configure logging (good practice)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a scheduler instance (global for the application)
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application startup and shutdown events.
    Used to start and stop the APScheduler.
    """
    logger.info("Starting scheduler...")

    # Add your jobs here, referencing the imported functions
    # Example: Run 'my_scheduled_task' every minute
    scheduler.add_job(my_scheduled_task, CronTrigger(minute="*"), id="my_periodic_job")
    scheduler.start()
    logger.info("Scheduler started.")
    yield  # Application runs here
    logger.info("Shutting down scheduler...")
    scheduler.shutdown()
    logger.info("Scheduler shut down.")

# Initialize your FastAPI app with the lifespan context manager
app = FastAPI(
    title="My Hackathon API with Scheduled Jobs",
    description="A basic API demonstrating scheduled background tasks.",
    version="1.0.0",
    lifespan=lifespan # Link the lifespan to your app
)

# --- Your regular API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "FastAPI with APScheduler running and tasks in 'util'!"}

# Example of another endpoint
@app.get("/status")
async def get_status():
    jobs = [{"id": job.id, "next_run_time": job.next_run_time.strftime("%Y-%m-%d %H:%M:%S") if job.next_run_time else "N/A"} for job in scheduler.get_jobs()]
    return {"status": "running", "scheduled_jobs": jobs}