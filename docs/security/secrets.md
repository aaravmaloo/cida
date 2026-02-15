# Secrets and Rotation

## Backend (Railway)

- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET`
- `ADMIN_PASSKEY_HASH` (Argon2)
- `SENTRY_DSN` (optional)
- `R2_ENDPOINT` (optional)
- `R2_BUCKET` (optional)
- `R2_ACCESS_KEY` (optional)
- `R_SEC2RET_KEY` (optional)

## Frontend (Vercel)

- `NEXT_PUBLIC_API_BASE_URL`

## Rotation policy

1. Rotate `JWT_SECRET` every 90 days.
2. Rotate `ADMIN_PASSKEY_HASH` immediately after suspected compromise.
3. Rotate object storage keys every 90 days.
4. Validate deploy health after each rotation using `/healthz` and `/readyz`.

