#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deploy the Nanopore PNP Neural-Operator portal to https://pino.statwhy.com
#
# Topology (isolated from every other site on the shared host):
#   * static portal  -> /var/www/pino.statwhy.com/   (contents of web_simulator/)
#   * live SG/FV API -> systemd unit pino-statwhy.service on 127.0.0.1:18765
#   * nginx vhost    -> /etc/nginx/sites-available/pino.statwhy.com
#                       serves the static tree and proxies /api/ to the unit
#   * TLS            -> certbot --nginx (auto redirect HTTP->HTTPS)
#
# Safety: only writes our own files, never modifies other vhosts, runs
# `nginx -t` before every reload, and verifies the service on loopback before
# touching nginx.  Idempotent: safe to re-run.
#
# Usage:
#   scripts/deploy_pino_statwhy.sh [--skip-certbot] [--dry-run]
# ---------------------------------------------------------------------------
set -euo pipefail

SSH_USER="${SSH_USER:-root}"
SSH_HOST="${SSH_HOST:-statwhy.com}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/kras_nanopore_codex}"
DOMAIN="${DOMAIN:-pino.statwhy.com}"
APP_DIR="${APP_DIR:-/opt/pino-statwhy}"
WEB_ROOT="${WEB_ROOT:-/var/www/pino.statwhy.com}"
SERVICE="${SERVICE:-pino-statwhy}"
PORT="${PORT:-18765}"

SKIP_CERTBOT=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --skip-certbot) SKIP_CERTBOT=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d /tmp/pino-stage.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

SSH_COMMON=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -i "$SSH_KEY")
remote() { ssh "${SSH_COMMON[@]}" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "$@"; }

echo "==> staging portal tree"
mkdir -p "$STAGE/app/scripts"
cp -R "$ROOT/web_simulator" "$STAGE/app/web_simulator"
cp "$ROOT/scripts/serve_interactive.py" "$STAGE/app/scripts/"
cp "$ROOT/scripts/generate_parametric_dataset.py" "$STAGE/app/scripts/"
cp -R "$ROOT/scripts/sg_baseline" "$STAGE/app/scripts/sg_baseline"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

if [ "$DRY_RUN" = "1" ]; then
  echo "==> dry run; staged at $STAGE (kept)"; trap - EXIT; exit 0
fi

echo "==> packing"
TARBALL="$STAGE/pino-portal.tgz"
tar -C "$STAGE/app" -czf "$TARBALL" .

echo "==> uploading to $SSH_USER@$SSH_HOST"
scp "${SSH_COMMON[@]}" -P "$SSH_PORT" "$TARBALL" "$SSH_USER@$SSH_HOST:/tmp/pino-portal.tgz"

echo "==> remote provisioning"
remote DOMAIN="$DOMAIN" APP_DIR="$APP_DIR" WEB_ROOT="$WEB_ROOT" SERVICE="$SERVICE" PORT="$PORT" SKIP_CERTBOT="$SKIP_CERTBOT" bash -s <<'REMOTE'
set -euo pipefail

echo "-- directories"
mkdir -p "$APP_DIR" "$WEB_ROOT"
rm -rf "$APP_DIR"/* "$APP_DIR"/.[!.]* 2>/dev/null || true
tar -C "$APP_DIR" -xzf /tmp/pino-portal.tgz
rm -f /tmp/pino-portal.tgz
chown -R www-data:www-data "$APP_DIR"

echo "-- publish static portal"
rm -rf "${WEB_ROOT:?}"/*
cp -R "$APP_DIR/web_simulator/." "$WEB_ROOT/"
chown -R www-data:www-data "$WEB_ROOT"

echo "-- systemd unit"
cat >/etc/systemd/system/${SERVICE}.service <<UNIT
[Unit]
Description=Nanopore PNP Neural-Operator live SG/FV API
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/python3 $APP_DIR/scripts/serve_interactive.py --host 127.0.0.1 --port $PORT
Restart=on-failure
RestartSec=3
User=www-data
Group=www-data
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now ${SERVICE}.service
sleep 2
systemctl --no-pager --lines=0 status ${SERVICE}.service || true

echo "-- wait for loopback API"
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/api/health"; echo
curl -fsS -X POST "http://127.0.0.1:${PORT}/api/solve_sg" \
  -H 'Content-Type: application/json' -d '{"V_app":50,"C0":100,"qw":-0.01,"R":5.0}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("loopback SG/FV mean_I pA =", round(d["mean_current_pA"],6), "residual", d["residual"], "iters", d["iterations"])'

echo "-- nginx vhost"
cat >/etc/nginx/sites-available/${DOMAIN} <<NGINX
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    root ${WEB_ROOT};
    index index.html;

    client_max_body_size 8m;

    location /api/ {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
    }

    location / {
        try_files \$uri \$uri/ =404;
    }
}
NGINX
ln -sf /etc/nginx/sites-available/${DOMAIN} /etc/nginx/sites-enabled/${DOMAIN}
nginx -t
systemctl reload nginx

if [ "${SKIP_CERTBOT}" != "1" ]; then
  echo "-- TLS via certbot"
  certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos --redirect --keep-until-expiring --no-eff-email \
    || certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos --redirect --keep-until-expiring
  nginx -t
  systemctl reload nginx
fi

echo "-- remote provisioning complete"
REMOTE

echo "==> verifying over HTTPS"
curl -fsSI "https://${DOMAIN}/" | head -1
curl -fsS "https://${DOMAIN}/" | grep -qi "Neural-Operator" && echo "OK  portal HTML served"
curl -fsS -X POST "https://${DOMAIN}/api/solve_sg" \
  -H 'Content-Type: application/json' -d '{"V_app":50,"C0":100,"qw":-0.01,"R":5.0}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("PUBLIC SG/FV mean_I pA =", round(d["mean_current_pA"],6), "residual", d["residual"])'

echo "==> deployed: https://${DOMAIN}/"
