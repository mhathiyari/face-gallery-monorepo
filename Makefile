.PHONY: install test test-backend test-frontend lint run docker-gpu docker-cpu

install:
	pip install -e backend/[dev,cpu]
	pip install -r frontend/requirements.txt
	pip install -r requirements-dev.txt

test: test-backend test-frontend

test-backend:
	cd backend && python -m pytest tests/unit/ -v

test-frontend:
	python -m pytest tests/test_frontend_smoke.py -v

lint:
	ruff check backend/src/ frontend/app.py
	black --check backend/src/ frontend/app.py

run:
	python frontend/app.py

docker-gpu:
	docker compose -f docker-compose.yml up --build

docker-cpu:
	docker compose -f docker-compose.cpu.yml up --build
