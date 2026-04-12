#!/bin/sh
set -e

MISSING=""

for var in OPENAI_API_KEY OPENAI_API_BASE; do
  eval val=\$$var
  if [ -z "$val" ]; then
    MISSING="$MISSING $var"
  fi
done

if [ -n "$MISSING" ]; then
  echo "{\"level\":\"FATAL\",\"module\":\"preflight\",\"message\":\"Missing required env vars:$MISSING\"}" >&2
  exit 1
fi

echo '{"level":"INFO","module":"preflight","message":"Pre-flight check passed"}'

exec "$@"
