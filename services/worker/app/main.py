import asyncio

from app.jobs import generate_report


if __name__ == "__main__":
    asyncio.run(generate_report({}, "sample", "sample"))

