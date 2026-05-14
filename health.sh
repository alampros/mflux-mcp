#!/usr/bin/env bash
# health.sh — project health check for agent-harness-kit
#
# Exit 0 = healthy, agents may proceed.
# Exit 1 = unhealthy, agents must stop and report.

set -euo pipefail

FAIL=0
WARN=0

pass() { printf "  ✓ %s\n" "$1"; }
fail() { printf "  ✗ %s\n" "$1"; FAIL=$((FAIL + 1)); }
warn() { printf "  ~ %s\n" "$1"; WARN=$((WARN + 1)); }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== mflux-mcp health check ==="

# ── 1. Python toolchain ─────────────────────────────────────────────
echo ""
echo "Toolchain:"
if command -v uv &>/dev/null; then
  pass "uv found: $(uv --version 2>&1)"
else
  fail "uv not found — install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if command -v python3 &>/dev/null; then
  pass "python3 found: $(python3 --version 2>&1)"
else
  fail "python3 not found"
fi

# ── 2. mflux installed as uv tool ───────────────────────────────────
echo ""
echo "mflux:"
if command -v mflux-generate-flux2 &>/dev/null; then
  mflux_version=$(uv tool list 2>/dev/null | grep '^mflux ' | head -1 || echo "unknown")
  pass "mflux installed: $mflux_version"
else
  fail "mflux not installed as uv tool — install via: uv tool install --upgrade mflux"
fi

# ── 3. pyproject.toml present ───────────────────────────────────────
echo ""
echo "Project:"
if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
  pass "pyproject.toml found"
else
  warn "pyproject.toml not yet created"
fi

if [[ -f "$SCRIPT_DIR/server.py" ]]; then
  pass "server.py found"
else
  warn "server.py not yet created"
fi

# ── 4. Tests ────────────────────────────────────────────────────────
echo ""
echo "Tests:"
if [[ -d "$SCRIPT_DIR/tests" ]]; then
  test_count=$(find "$SCRIPT_DIR/tests" -name 'test_*.py' | wc -l | tr -d ' ')
  if [[ "$test_count" -gt 0 ]]; then
    pass "tests/ contains $test_count test file(s)"
    if uv run pytest tests/ --tb=short -q 2>&1 | tail -1 | grep -qE "passed|no tests ran"; then
      pass "pytest passes"
    else
      fail "pytest has failures"
    fi
  else
    warn "tests/ exists but contains no test_*.py files"
  fi
else
  warn "tests/ directory not found yet"
fi

# ── 5. Docs present ─────────────────────────────────────────────────
echo ""
echo "Docs:"
if [[ -f "$SCRIPT_DIR/PLAN.md" ]]; then
  pass "PLAN.md found"
else
  warn "PLAN.md not found"
fi

# ── 6. Feature list ─────────────────────────────────────────────────
echo ""
echo "Harness:"
FL="$SCRIPT_DIR/.harness/feature_list.json"
if [[ -f "$FL" ]]; then
  feature_count=$(python3 -c "import json; print(len(json.load(open('$FL'))))" 2>/dev/null || echo "0")
  if [[ "$feature_count" -gt 0 ]]; then
    pass "feature_list.json has $feature_count feature(s)"
  else
    fail "feature_list.json is empty"
  fi
else
  fail "feature_list.json not found"
fi

# ── 7. Apple Silicon check (mflux requires MLX) ─────────────────────
echo ""
echo "Hardware:"
if [[ "$(uname -m)" == "arm64" ]]; then
  pass "Apple Silicon detected (required for MLX/mflux)"
else
  fail "Not Apple Silicon — mflux requires Apple Silicon (arm64) for MLX"
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
if [[ "$FAIL" -gt 0 ]]; then
  echo "UNHEALTHY: $FAIL failure(s), $WARN warning(s)"
  exit 1
else
  echo "HEALTHY: 0 failures, $WARN warning(s)"
  exit 0
fi
