#!/bin/bash
# embed-pro status dashboard
# Usage: ./status.sh [host:port]

URL="${1:-http://localhost:8020}"
BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
DIM="\033[2m"
RESET="\033[0m"

health=$(curl -sf "$URL/health" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo -e "${RED}${BOLD}  embed-pro is NOT running${RESET} ($URL)"
    exit 1
fi

status=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
uptime=$(echo "$health" | python3 -c "import sys,json; s=json.load(sys.stdin)['uptime_seconds']; d=s//86400; h=(s%86400)//3600; m=(s%3600)//60; print(f'{d}d {h}h {m}m' if d else f'{h}h {m}m' if h else f'{m}m {s%60}s')")
embed=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['embed_loaded'])")
rerank=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['rerank_loaded'])")
embed_req=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['embed_requests'])")
rerank_req=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['rerank_requests'])")
inflight=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['inflight_requests'])")
ws=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['ws_connections'])")
cache_size=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['cache_size'])")
cache_cap=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['cache_capacity'])")
mem=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['memory_mb'])")
backend=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['backend'])")
device=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['device'])")

if [ "$status" = "ok" ]; then
    status_color=$GREEN
elif [ "$status" = "draining" ]; then
    status_color=$YELLOW
else
    status_color=$RED
fi

embed_icon=$( [ "$embed" = "True" ] && echo "${GREEN}loaded${RESET}" || echo "${DIM}not loaded${RESET}")
rerank_icon=$( [ "$rerank" = "True" ] && echo "${GREEN}loaded${RESET}" || echo "${DIM}not loaded${RESET}")

echo ""
echo -e "  ${BOLD}embed-pro${RESET}  ${status_color}${status}${RESET}  ${DIM}uptime ${uptime}${RESET}"
echo ""
echo -e "  ${CYAN}Models${RESET}"
echo -e "    bge-m3            $embed_icon  ${DIM}($backend/$device)${RESET}"
echo -e "    bge-reranker-v2   $rerank_icon  ${DIM}($device)${RESET}"
echo ""
echo -e "  ${CYAN}Traffic${RESET}"
echo -e "    Embed requests    ${BOLD}$embed_req${RESET}"
echo -e "    Rerank requests   ${BOLD}$rerank_req${RESET}"
echo -e "    In-flight         ${BOLD}$inflight${RESET}"
echo -e "    WebSocket conns   ${BOLD}$ws${RESET}"
echo ""
echo -e "  ${CYAN}Resources${RESET}"
echo -e "    Memory            ${BOLD}${mem} MB${RESET}"
echo -e "    Cache             ${BOLD}${cache_size}${RESET} / ${cache_cap}"
echo ""
