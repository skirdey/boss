from flask import Flask, request, jsonify, render_template, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_wtf.csrf import CSRFProtect
from datetime import datetime
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with your secret key
csrf = CSRFProtect(app)

# MongoDB Configuration
try:
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['task_db']
    tasks_collection = db['tasks']
except ServerSelectionTimeoutError as err:
    logger.error(f"Error: Could not connect to MongoDB server: {err}")
    client = None
    tasks_collection = None

def is_db_available():
    return client is not None and tasks_collection is not None

@app.route('/')
def index():
    if not is_db_available():
        return render_template('db_error.html'), 503
    try:
        # Exclude deleted tasks
        tasks = list(tasks_collection.find({"is_deleted": {"$ne": True}}).sort('created_at', -1))
        for task in tasks:
            task['_id'] = str(task['_id'])
        return render_template('index.html', tasks=tasks)
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return render_template('db_error.html'), 500


@app.route('/tasks/<task_id>')
def task_detail(task_id):
    if not is_db_available():
        return render_template('db_error.html'), 503
    try:
        # Exclude deleted tasks
        task = tasks_collection.find_one({'_id': ObjectId(task_id), "is_deleted": {"$ne": True}})
        if not task:
            return render_template('404.html'), 404
        task['_id'] = str(task['_id'])
        return render_template('task_detail.html', task=task)
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return render_template('db_error.html'), 500


@app.route('/tasks/<task_id>/delete', methods=['POST'])
def delete_task(task_id):
    if not is_db_available():
        return render_template('db_error.html'), 503
    try:
        result = tasks_collection.update_one(
            {'_id': ObjectId(task_id)},
            {'$set': {'is_deleted': True, 'updated_at': datetime.utcnow()}}
        )
        if result.matched_count == 0:
            return render_template('404.html'), 404
        return redirect(url_for('index'))
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return render_template('db_error.html'), 500


@app.route('/new-task', methods=['GET', 'POST'])
def new_task():
    if not is_db_available():
        return render_template('db_error.html'), 503
    if request.method == 'POST':
        description = request.form.get('description')
        task_type = request.form.get('type', 'general')
        evaluation_criteria = request.form.get('evaluation_criteria', '')
        max_retries = int(request.form.get('max_retries', 3))

        if not description:
            return "Description is required", 400

        task = {
            "description": description,
            "type": task_type,
            "status": "Created",
            "evaluation_criteria": evaluation_criteria,
            "retry_count": 0,
            "max_retries": max_retries,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        try:
            result = tasks_collection.insert_one(task)
            return redirect(url_for('task_detail', task_id=str(result.inserted_id)))
        except PyMongoError as err:
            logger.error(f"Database error: {err}")
            return render_template('db_error.html'), 500
    else:
        return render_template('new_task.html')

@app.route('/api/tasks', methods=['POST'])
def create_task():
    if not is_db_available():
        return jsonify({'error': 'Database unavailable'}), 503
    data = request.json
    if not data or 'description' not in data:
        return jsonify({'error': 'Task description is required'}), 400
    task = {
        "description": data['description'],
        "type": data.get('type', 'general'),
        "status": "Created",
        "evaluation_criteria": data.get('evaluation_criteria', ''),
        "retry_count": 0,
        "max_retries": data.get('max_retries', 3),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    try:
        result = tasks_collection.insert_one(task)
        task_id = str(result.inserted_id)
        return jsonify({'message': 'Task created', 'task_id': task_id}), 201
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return jsonify({'error': 'Database error occurred'}), 500


@app.route('/reset-tasks', methods=['POST'])
def reset_tasks():
    if not is_db_available():
        return render_template('db_error.html'), 503
    try:
        result = tasks_collection.update_many(
            {},  # Reset all tasks
            {'$set': {'status': 'Created', 'updated_at': datetime.utcnow()}}
        )
        return redirect(url_for('index'))
    except PyMongoError as err:
        logger.error(f"Database error: {err}")
        return render_template('db_error.html'), 500


# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.errorhandler(503)
def service_unavailable_error(error):
    return render_template('db_error.html'), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

