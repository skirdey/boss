import logging
import os
from datetime import datetime, timezone

from bson.objectid import ObjectId
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from wtforms import TextAreaField
from wtforms.validators import DataRequired, Length, Optional

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

        return render_template("task_detail.html", task=task)
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return render_template("db_error.html"), 500


@app.route("/tasks/<task_id>/delete", methods=["POST"])
def delete_task(task_id):
    if not is_db_available():
        return render_template("db_error.html"), 503
    try:
        result = tasks_collection.delete_one(
            {"_id": ObjectId(task_id)},
        )
        if result.matched_count == 0:
            return render_template("404.html"), 404
        return redirect(url_for("index"))
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return render_template("db_error.html"), 500


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


@app.route("/new-task", methods=["GET", "POST"])
def new_task():
    if not is_db_available():
        return render_template("db_error.html"), 503

    form = TaskForm()

    if form.validate_on_submit():
        # Prepare step_estimation data

        task = {
            "description": form.description.data,
            "status": "Created",
            "evaluation_criteria": form.evaluation_criteria.data,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "steps": [],
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
    task = {
        "description": data["description"],
        "status": "Created",
        "evaluation_criteria": data.get("evaluation_criteria", ""),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "steps": [],
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
        tasks_collection.update_many(
            {},
            {
                "$set": {
                    "status": "Created",
                    "current_agent": None,
                    "current_step_index": 0,
                    "notes": "",
                    "final_failure_details": {},
                    "updated_at": current_time,
                    "steps": [],
                    "audit_trail": [],
                    "finished_at": None,
                },
                "$unset": {
                    "result": "",
                    "final_failure_details": {},
                    "human_intervention_request": {},
                    "failure_history": [],
                },  # Remove the 'result' field if necessary
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
    app.run(host="0.0.0.0", debug=True)
