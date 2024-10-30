import logging
import os
from datetime import datetime, timezone

from bson.objectid import ObjectId
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from wtforms import BooleanField, IntegerField, TextAreaField
from wtforms.validators import DataRequired, Length, NumberRange, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "your-secret-key"  # Replace with your secret key
csrf = CSRFProtect(app)

# MongoDB Configuration
try:
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["task_db"]
    tasks_collection = db["tasks"]
except ServerSelectionTimeoutError as err:
    logger.error(f"Error: Could not connect to MongoDB server: {err}")
    client = None
    tasks_collection = None


def is_db_available():
    return client is not None and tasks_collection is not None


def map_status_to_css(status):
    status_mapping = {
        "Created": "created",
        "Running": "running",
        "Failed": "failed",
        "Completed_Workflow": "completed",
        "Pending": "pending",
        # Add other mappings as needed
    }
    return status_mapping.get(status, "unknown")


@app.route("/")
def index():
    if not is_db_available():
        return render_template("db_error.html"), 503
    try:
        tasks_cursor = tasks_collection.find({"is_deleted": {"$ne": True}}).sort(
            "created_at", -1
        )
        tasks = []
        for task in tasks_cursor:
            task["_id"] = str(task["_id"])
            task["created_at"] = task["created_at"]
            task["updated_at"] = task["updated_at"]
            task["css_status"] = map_status_to_css(task["status"])
            tasks.append(task)

        return render_template("index.html", tasks=tasks)
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return render_template("db_error.html"), 500


@app.route("/tasks/<task_id>")
def task_detail(task_id):
    if not is_db_available():
        return render_template("db_error.html"), 503
    try:
        # Exclude deleted tasks
        task = tasks_collection.find_one(
            {"_id": ObjectId(task_id), "is_deleted": {"$ne": True}}
        )
        if not task:
            return render_template("404.html"), 404

        # Ensure step_estimation exists
        if "step_estimation" not in task:
            task["step_estimation"] = {
                "estimated_steps": [],
                "total_estimated_duration": 30,
                "estimation_timestamp": datetime.now(timezone.utc),
                "estimation_model_version": "gpt-4o",
                "estimation_confidence": 0.85,
            }
        else:
            # Ensure 'estimation_confidence' exists within 'step_estimation'
            if "estimation_confidence" not in task["step_estimation"]:
                task["step_estimation"]["estimation_confidence"] = 0.85
                # Optionally, update the database to reflect this change
                tasks_collection.update_one(
                    {"_id": task["_id"]},
                    {"$set": {"step_estimation.estimation_confidence": 0.85}},
                )

        # Ensure workflow_state exists
        if "workflow_state" not in task:
            task["workflow_state"] = {
                "completed_steps": [],
                "remaining_steps": [],
                "current_agent": None,
            }
        else:
            # Ensure 'completed_steps' and 'remaining_steps' exist
            task["workflow_state"].setdefault("completed_steps", [])
            task["workflow_state"].setdefault("remaining_steps", [])

        # Convert ObjectId to string for template usage
        task["_id"] = str(task["_id"])

        # Format datetime fields for better readability
        task["created_at"] = (
            task["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            if task.get("created_at")
            else "N/A"
        )
        task["updated_at"] = (
            task["updated_at"].strftime("%Y-%m-%d %H:%M:%S")
            if task.get("updated_at")
            else "N/A"
        )
        task["step_estimation"]["estimation_timestamp"] = (
            task["step_estimation"]["estimation_timestamp"].strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if task["step_estimation"].get("estimation_timestamp")
            else "N/A"
        )

        return render_template("task_detail.html", task=task)
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return render_template("db_error.html"), 500


@app.route("/tasks/<task_id>/delete", methods=["POST"])
def delete_task(task_id):
    if not is_db_available():
        return render_template("db_error.html"), 503
    try:
        current_time = datetime.now(timezone.utc)
        result = tasks_collection.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": {"is_deleted": True, "updated_at": current_time}},
        )
        if result.matched_count == 0:
            return render_template("404.html"), 404
        return redirect(url_for("index"))
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return render_template("db_error.html"), 500


def add_missing_fields():
    if not is_db_available():
        logger.error("Database unavailable. Cannot add missing fields.")
        return
    try:
        tasks_collection.update_many(
            {"current_step": {"$exists": False}},
            {
                "$set": {
                    "current_step": 0,
                    "estimated_total_steps": 1,  # Default to 1; adjust as needed
                    "workflow_state": {
                        "completed_steps": [],
                        "remaining_steps": [],
                        "current_agent": None,
                    },
                    "step_estimation": {
                        "estimated_steps": [],
                        "total_estimated_duration": 30,  # Default to 30 minutes
                        "estimation_timestamp": datetime.now(timezone.utc),
                        "estimation_model_version": "gpt-4o",
                        "estimation_confidence": 0.85,
                    },
                    "is_continuous": False,
                    "audit_trail": [],
                }
            },
        )
        logger.info("Missing fields added to existing tasks.")
    except PyMongoError as err:
        logger.error(f"Error updating tasks: {err}")


class TaskForm(FlaskForm):
    description = TextAreaField(
        "Description",
        validators=[
            DataRequired(message="Description is required"),
            Length(
                min=5,
                max=200,
                message="Description must be between 5 and 200 characters",
            ),
        ],
        description="Enter a clear description of the task (5-200 characters)",
    )

    evaluation_criteria = TextAreaField(
        "Evaluation Criteria",
        validators=[Optional()],
        description="Optional: Define success criteria for this task",
    )

    max_retries = IntegerField(
        "Max Retries",
        validators=[
            NumberRange(min=0, max=10, message="Max retries must be between 0 and 10")
        ],
        default=3,
        description="Number of retry attempts allowed (0-10)",
    )

    estimated_duration = IntegerField(
        "Estimated Duration (minutes)",
        validators=[
            Optional(),
            NumberRange(min=1, message="Duration must be at least 1 minute"),
        ],
        default=30,
        description="Estimated time to complete the task in minutes",
    )

    is_continuous = BooleanField(
        "Is Continuous", default=False, description="Is this task continuous?"
    )


@app.route("/new-task", methods=["GET", "POST"])
def new_task():
    if not is_db_available():
        return render_template("db_error.html"), 503

    form = TaskForm()

    if form.validate_on_submit():
        # Prepare step_estimation data
        step_estimation = {
            "estimated_steps": [],  # You can populate this based on task type or other logic
            "total_estimated_duration": form.estimated_duration.data
            or 30,  # default to 30 if not provided
            "estimation_timestamp": datetime.now(timezone.utc),
            "estimation_model_version": "gpt-4o",  # Update as per your model versioning
            "estimation_confidence": 0.85,  # Default confidence score
        }

        # Prepare workflow_state data
        workflow_state = {
            "completed_steps": [],
            "remaining_steps": [],  # Populate based on step_estimation
            "current_agent": None,
        }

        task = {
            "description": form.description.data,
            "status": "Created",
            "evaluation_criteria": form.evaluation_criteria.data,
            "retry_count": 0,
            "max_retries": form.max_retries.data,
            "is_deleted": False,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "estimated_total_steps": 1,  # Adjust based on step_estimation
            "current_step": 0,
            "workflow_state": workflow_state,
            "step_estimation": step_estimation,
            "is_continuous": form.is_continuous.data,
            "audit_trail": [],  # Initialize empty audit trail
        }

        try:
            result = tasks_collection.insert_one(task)
            return redirect(url_for("task_detail", task_id=str(result.inserted_id)))
        except PyMongoError as err:
            logger.error(f"Database error: {err}")
            return render_template("db_error.html"), 500

    return render_template("new_task.html", form=form)


@app.route("/api/tasks", methods=["POST"])
def create_task():
    if not is_db_available():
        return jsonify({"error": "Database unavailable"}), 503
    data = request.json
    if not data or "description" not in data:
        return jsonify({"error": "Task description is required"}), 400

    # Prepare step_estimation data
    step_estimation = {
        "estimated_steps": [],  # Populate based on task type or logic
        "total_estimated_duration": data.get("estimated_duration", 30),
        "estimation_timestamp": datetime.now(timezone.utc),
        "estimation_model_version": "gpt-4o",
        "estimation_confidence": 0.85,
    }

    # Prepare workflow_state data
    workflow_state = {
        "completed_steps": [],
        "remaining_steps": [],
        "current_agent": None,
    }

    task = {
        "description": data["description"],
        "status": "Created",
        "evaluation_criteria": data.get("evaluation_criteria", ""),
        "retry_count": 0,
        "max_retries": data.get("max_retries", 3),
        "is_deleted": False,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "estimated_total_steps": 1,  # Adjust as needed
        "current_step": 0,
        "workflow_state": workflow_state,
        "step_estimation": step_estimation,
        "is_continuous": data.get("is_continuous", False),
        "audit_trail": [],
    }
    try:
        result = tasks_collection.insert_one(task)
        task_id = str(result.inserted_id)
        return jsonify({"message": "Task created", "task_id": task_id}), 201
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return jsonify({"error": "Database error occurred"}), 500


@app.route("/reset-tasks", methods=["POST"])
def reset_tasks():
    if not is_db_available():
        return render_template("db_error.html"), 503
    try:
        current_time = datetime.now(timezone.utc)
        result = tasks_collection.update_many(
            {"is_deleted": {"$ne": True}},  # Only reset tasks that are not deleted
            {
                "$set": {
                    "status": "Created",
                    "updated_at": current_time,
                    "retry_count": 0,
                    "current_step": 0,
                    "workflow_state.completed_steps": [],
                    "workflow_state.remaining_steps": [],
                    "workflow_state.current_agent": None,
                    "step_estimation.remaining_duration": "$step_estimation.total_estimated_duration",
                    # Reset other fields as necessary
                },
                "$unset": {"result": ""},  # Remove the 'result' field if necessary
            },
        )
        return jsonify({"message": "Tasks reset successfully"}), 200
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return jsonify({"error": "Database error occurred"}), 500


@app.route("/api/tasks/<task_id>", methods=["PUT"])
def update_task(task_id):
    if not is_db_available():
        return jsonify({"error": "Database unavailable"}), 503
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Update 'updated_at' timestamp
    data["updated_at"] = datetime.now(timezone.utc)

    try:
        result = tasks_collection.update_one(
            {"_id": ObjectId(task_id), "is_deleted": {"$ne": True}}, {"$set": data}
        )
        if result.matched_count == 0:
            return jsonify({"error": "Task not found"}), 404
        return jsonify({"message": "Task updated successfully"}), 200
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return jsonify({"error": "Database error occurred"}), 500


def migrate_tasks_add_step_estimation():
    if not is_db_available():
        return
    try:
        tasks_collection.update_many(
            {"step_estimation": {"$exists": False}},
            {
                "$set": {
                    "step_estimation": {
                        "estimated_steps": [],
                        "total_estimated_duration": 30,  # default 30 minutes
                        "estimation_timestamp": datetime.now(timezone.utc),
                        "estimation_model_version": "gpt-4o",
                        "estimation_confidence": 0.85,
                    }
                }
            },
        )
        logger.info("Successfully added step_estimation to existing tasks")
    except PyMongoError as err:
        logger.error(f"Error migrating tasks: {err}")


# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template("500.html"), 500


@app.errorhandler(503)
def service_unavailable_error(error):
    return render_template("db_error.html"), 503


if __name__ == "__main__":
    add_missing_fields()
    migrate_tasks_add_step_estimation()
    app.run(host="0.0.0.0", debug=True)
