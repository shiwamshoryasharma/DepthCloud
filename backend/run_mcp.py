"""Start the read-only MCP stdio server; run the HTTP backend separately."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from depthcloud.integrations.mcp_server import server
    server.run(transport="stdio")

