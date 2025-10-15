# Model Zoo Registry API

This document explains how to use the FastAPI-based registry service that ships with the project. It covers architecture, local development, Docker deployment, and HTTP usage examples.

## Overview

The API exposes endpoints for:

- Registering a model configuration (hyperparameters, variable transforms, evaluation spec) with environment scoping.
- Persisting model specificatidocker compose ons (id, version, hyperparameter schema) independently of configurations.
- Retrieving model specs and configuration history from PostgreSQL.

Under the hood:

- `src/model_zoo/api/app.py` defines the FastAPI application.
- `RegistryDatabase` manages two tables: `model_specs` and `model_configs`.
- `load_model_configuration_from_dict` validates incoming JSON configs using existing configuration tooling.

## Architecture Diagram

```
Client → FastAPI (uvicorn) → RegistryDatabase (psycopg) → PostgreSQL
                                      ↓
                              ModelRegistry (in-memory)
```

## Prerequisites

- Docker 20+ and Docker Compose plugin.
- Python 3.11 if you want to run locally without containers.

## Quick Start (Docker Compose)

1. Build and launch the stack:

   ```bash
   docker compose up --build
   ```

   This starts two services:

   - `postgres` (container `model-zoo-postgres`): PostgreSQL 15 with health checks.
   - `api` (container `model-zoo-api`): FastAPI service reachable at `http://localhost:8000`.

2. Verify health:

   ```bash
   curl http://localhost:8000/health
   # {"status":"ok"}
   ```

3. Register a configuration (example uses the bundled linear regression definition):

   ```bash
   curl -X POST http://localhost:8000/registry/configs \
        -H "Content-Type: application/json" \
         -d @<(jq -n --argfile cfg configs/linear_regression.json \
             '{env:"dev", config:$cfg}')
   ```

   Or run the scripted helper:

   ```bash
   python examples/register_model_via_api.py --config configs/linear_regression.json --env dev
   ```

   For a fully commented walkthrough (marketing mix example), open `docs/notebooks/marketing_mix_registry_walkthrough.ipynb` in Jupyter. A dedicated table-loading demo lives at `docs/notebooks/config_table_loader_demo.ipynb`.

4. List registered models:

   ```bash
   curl http://localhost:8000/registry/models | jq
   ```

5. Fetch a single model (with configs):

   ```bash
   curl http://localhost:8000/registry/models/linear_regression_v1 | jq
   ```

6. Submit a run using the stored configuration (synthetic linear data). The
   endpoint is asynchronous and returns immediately with a run ticket:

   ```bash
   curl -X POST http://localhost:8000/runs \
        -H "Content-Type: application/json" \
        -d '{
              "model_id": "linear_regression_v1",
              "env": "dev",
              "data_sources": {
                "x_values": [0.1, 0.2, 0.3, 0.4, 0.5],
                "y_values": [1.2, 1.4, 1.6, 1.8, 2.0]
              },
              "evaluation": {
                "metric_ids": ["r2_v1", "mse_v1"],
                "y_true_key": "y"
              }
            }'
   # => {"run_id": "...", "status": "queued"}
   ```

   Poll the ticket using `GET /runs/{run_id}` (the helper script handles this):

   ```bash
   curl http://localhost:8000/runs/<run_id> | jq
   ```

   Once the run completes the payload includes diagnostics, evaluation metrics,
   and the rendered summary string. You can also use the helper script:

   ```bash
   python examples/register_model_via_api.py --config configs/linear_regression.json --env dev --run
   ```

7. Tear down containers:

   ```bash
   docker compose down -v
   ```

## Running the API Locally (without Docker)

1. Install dependencies:

   ```bash
   pip install -e .
   ```

2. Provide a PostgreSQL instance (e.g., via Docker as above) and set credentials:

   ```bash
   export POSTGRES_DSN=postgresql://model_zoo_user:change_me@localhost:5432/model_zoo
   ```

   Alternatively, export `POSTGRES_HOST`, `POSTGRES_USER`, etc. or populate `.env` (the API reads `MODEL_ZOO_ENV_FILE`).

3. Start uvicorn:

   ```bash
   uvicorn model_zoo.api.app:app --reload
   ```

## Configuration Payload Structure

A configuration JSON must include the following sections:

- `model`: id, name, version, kind (`library`/`api`/`pipeline`), optional tags/metadata.
- `hyperparameters`: `schema` (list of rows) and `values` (engine table).
  Optional `tables` entries let you pull additional rows from CSV/JSON files.
- `coefficient_priors`: intercept + per-variable priors (`distribution`, nested
  `params`, and optional `constraints`) kept separate from runtime
  hyperparameters. Priors can be supplied inline or via CSV/JSON tables.
  Constraints support `sign` (`positive`, `negative`, `nonnegative`, etc.) and
  `lower` / `upper` bounds for truncated priors.
- `variables`: `definitions` (source, transforms) and optional `groups`.
- `evaluation`: metric ids, target key, split options.
- `context`: execution hints (timeout, device, resources).

Use `configs/linear_regression.json` as a template. The API validates types and ensures the hyperparameter schema is coherent.

## Endpoints Summary

| Method | Path                               | Description                                  |
|--------|------------------------------------|----------------------------------------------|
| GET    | `/health`                          | Service heartbeat                            |
| POST   | `/registry/configs`                | Register config JSON (auto-upserts model spec)|
| POST   | `/registry/models`                 | Register a model spec directly                |
| GET    | `/registry/models`                 | List all registered model specs               |
| GET    | `/registry/models/{model_id}`      | Fetch spec plus associated configs            |
| GET    | `/registry/models/{model_id}/configs` | List configs (optionally filter by env)    |
| POST   | `/runs`                            | Submit a run (asynchronous, returns ticket)   |
| GET    | `/runs`                            | List run tickets and their status             |
| GET    | `/runs/{run_id}`                   | Fetch status/result for a submitted run       |

## Response Examples

### Register Config

```json
{
  "model_id": "linear_regression_v1",
  "environment": "dev",
  "spec": {
    "model_id": "linear_regression_v1",
    "name": "Linear Regression (Config)",
    "version": "1.0.0",
    "kind": "library",
    "hyperparams_schema": {"type": "object", ...},
    "tags": ["linear", "config-driven"],
    "metadata": {}
  },
  "stored": true
}
```

### List Models

```json
[
  {
    "model_id": "linear_regression_v1",
    "name": "Linear Regression (Config)",
    "version": "1.0.0",
    "kind": "library",
    "hyperparams": {"type": "object", ...},
    "tags": ["linear", "config-driven"],
    "metadata": {},
    "registered_at": "2024-06-10T12:34:56.123456+00:00",
    "updated_at": "2024-06-10T12:34:56.123456+00:00"
  }
]
```

## Extending the API

- Add authentication: wrap routes with FastAPI dependencies that verify JWT/Bearer tokens.
- Versioning: mount routers under `/v1`, `/v2` for backward compatibility.
- Additional tables: extend `RegistryDatabase` to track run history or binding definitions.
- Async support: migrate to `psycopg.AsyncConnection` and `FastAPI` async routes for higher concurrency.

## Troubleshooting

- **Connection refused**: Ensure the Postgres container is healthy (`docker compose ps`).
- **Credential errors**: Check the `.env` file or environment variables match the compose defaults.
- **Schema errors**: The API surfaces validation errors raised during configuration parsing; inspect the `detail` field in the 400 response.
- **Port conflicts**: Change published ports in `docker-compose.yml` or stop existing services using `lsof -i :8000`.

## Related Files

- `Dockerfile`: builds the API image with uvicorn entrypoint.
- `docker-compose.yml`: orchestrates Postgres + API services.
- `src/model_zoo/api/app.py`: route definitions and database helper.
- `configs/`: sample configuration payloads you can POST directly.

## Next Steps

- Integrate the API with CI/CD to deploy automatically after changes.
- Add unit tests for API routes using `fastapi.testclient` and temporary Postgres containers (e.g., `pytest-postgresql`).
- Provide SDK helpers that wrap the HTTP calls for Python clients.
