#!/bin/sh
# Prepares the compose Garage for the store's integration tests: a single-node
# layout, the bucket, and fixed development credentials so no secret has to be
# copied out of a log.
#
#   docker compose -f deploy/docker-compose.yml up -d garage
#   sh deploy/garage-init.sh
#
# Idempotent: every step tolerates already having been done.
set -eu

COMPOSE=${COMPOSE:-docker compose -f deploy/docker-compose.yml}
BUCKET=${BUCKET:-agent-store-test}
KEY_NAME=${KEY_NAME:-agent-dev}
# The format Garage expects: "GK" + 24 hex characters, and a 64-hex secret.
KEY_ID=${AGENT_STORE_S3_KEY:-GK0123456789abcdef01234567}
KEY_SECRET=${AGENT_STORE_S3_SECRET:-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef}

g() { $COMPOSE exec -T garage /garage "$@"; }

# A fresh node has no layout, and no layout means no storage.
NODE_ID=$(g node id -q | cut -d@ -f1)
if ! g layout show | grep -q "NO ROLE ASSIGNED" && g layout show | grep -q "$NODE_ID"; then
    : # already in the layout
else
    g layout assign -z dev -c 1G "$NODE_ID" || true
    # `layout apply` takes the version it is about to create, which is the current
    # one plus one.
    CURRENT=$(g layout show | sed -n 's/^Current cluster layout version: \([0-9]*\)$/\1/p' | tail -1)
    g layout apply --version "$((${CURRENT:-0} + 1))"
fi

g bucket create "$BUCKET" 2>/dev/null || true
g key import --yes -n "$KEY_NAME" "$KEY_ID" "$KEY_SECRET" 2>/dev/null || true
g bucket allow --read --write --owner "$BUCKET" --key "$KEY_NAME"

echo "garage ready: bucket=$BUCKET key=$KEY_ID"
