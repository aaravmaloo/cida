# Docker Local Stack

Run from repo root:

```bash
docker compose -f infra/docker/docker-compose.yml up --build
```

Services:

- Web: `http://localhost:3000`
- API: `http://localhost:8000`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`
