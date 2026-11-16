# Container Deployment

The backend image is platform-neutral. Build it from this directory with
`docker build -t aoa-backend .`. It listens on `PORT` (default `8000`) and
serves `GET /health` for health checks. Configure the required provider keys,
`AUTH_SECRET_KEY`, and a PostgreSQL `DATABASE_URL` in the container
environment. Set `CORS_ORIGINS` to the frontend's public origin.

Run schema upgrades as a one-off task for each release, before starting new
backend instances:

```sh
docker run --rm --env-file .env aoa-backend migrate upgrade head
```

Then run the backend, mapping the port assigned by your container host:

```sh
docker run --rm --env-file .env -e PORT=8000 -p 8000:8000 aoa-backend
```

ChromaDB uses `CHROMA_PERSIST_DIR` (default `.api_data/chroma`). Mount durable
storage there when using local Chroma. Local Chroma storage is not shared
between replicas; use a shared vector database for horizontally scaled
deployments. The upload directory only holds temporary files during ingestion.

The frontend image is built from `frontend-AOA-LLM/` with
`docker build -t aoa-frontend .`. At runtime it listens on `PORT` (default
`8080`) and reads `API_BASE_URL` and `GOOGLE_CLIENT_ID`. This lets the same
image move between environments without rebuilding. Set `API_BASE_URL` to the
public backend URL and include the frontend origin in backend `CORS_ORIGINS`.
TLS can terminate at the cloud load balancer or ingress.

The workspace's `aoa-deployment/` directory contains the Compose bundle. It
builds the images from the backend and frontend GitHub repositories, starts
PostgreSQL, runs migrations, and then starts both applications. See that
directory's README for local startup and environment setup. Cloud hosts can
also run each image with their own managed database, persistent storage,
networking, and secret configuration.
