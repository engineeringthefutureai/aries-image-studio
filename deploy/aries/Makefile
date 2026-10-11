.PHONY: all build up down restart logs health status metrics

all: up

PORT ?= 8000
RENDER_GID ?= $(shell getent group render 2>/dev/null | cut -d: -f3)
VIDEO_GID ?= $(shell getent group video 2>/dev/null | cut -d: -f3)
export RENDER_GID VIDEO_GID

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f --tail=100

health:
	curl -s http://localhost:$(PORT)/health | jq .

status:
	curl -s http://localhost:$(PORT)/status | jq .

metrics:
	curl -s http://localhost:$(PORT)/metrics
