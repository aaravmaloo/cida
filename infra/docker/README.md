# Docker Local Stack

Run from repo root:

```bash
docker compose -f infra/docker/docker-compose.yml up --build
```

If you need the full Torch-backed humanizer model in API, pass:

```bash
docker build -f infra/docker/Dockerfile.api --build-arg INSTALL_TORCH_CPU=1 .
```

Services:

- Web: `http://localhost:3000`
- API: `http://localhost:8000`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`
