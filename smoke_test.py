import sys

from config import load_from_env
from llm import create_client


def main() -> int:
    config = load_from_env()
    print("Loaded config:")
    print(f"  base_url={config.base_url or '<missing>'}")
    print(f"  model={config.model or '<missing>'}")

    try:
        client = create_client(
            base_url=config.base_url,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    except Exception as exc:
        print(f"\nClient creation failed: {exc}")
        return 1

    ok, message = client.healthcheck()
    print(f"\nHealth check: {'OK' if ok else 'FAILED'}")
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
