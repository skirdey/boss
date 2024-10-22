# boss
os for agent orchestration


### local


First create a virtual environment and install dependencies
```bash 
pip install -r requirements.txt
```

Build flask api and UI image 
```bash 
cd web && docker compose build
```

Start local services
```bash
# in the root directory
docker compose up
```
It starts local UI, kafka, mongo, and zookeeper

Start boss
```bash
./start.py
```
Starts agents and orchestrator
