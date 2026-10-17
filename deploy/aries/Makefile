.PHONY: all build up down restart logs metrics health

all: up

METRICS_PORT ?= 9464
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

metrics:
	curl -s http://localhost:$(METRICS_PORT)/metrics

health:
	curl -s http://localhost:$(METRICS_PORT)/healthz; curl -s -o /dev/null -w 'readyz: %{http_code}\n' http://localhost:$(METRICS_PORT)/readyz
