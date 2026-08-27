"""Project-root launcher for the Crowd Density web application."""

import socket

def _server_is_running(host: str, port: int) -> bool:
    """Avoid a second Uvicorn process trying to bind the same port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        return connection.connect_ex((host, port)) == 0


if __name__ == "__main__":
    import uvicorn

    if _server_is_running("127.0.0.1", 8000):
        print("Server already running at http://127.0.0.1:8000/")
    else:
        from backend.app import app

        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
