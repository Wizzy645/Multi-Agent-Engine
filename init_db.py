import asyncio
import os
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import AsyncPostgresSaver

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:password@db:5432/postgres")


async def initialize_memory(max_retries: int = 6, base_delay: float = 2.0):
    """
    Initialize LangGraph async checkpoint tables.
    
    Senior Note:
    - AsyncPostgresSaver + AsyncConnectionPool prevent blocking the FastAPI event loop.
    - dict_row is mandatory: the checkpointer accesses columns by name (dictionary-style).
    - autocommit=True is mandatory: DDL statements (CREATE TABLE) must commit immediately.
    - Exponential backoff handles the container race condition where Python boots before Postgres.
    """
    
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        min_size=1,
        max_size=10,
        kwargs={
            "row_factory": dict_row,
            "autocommit": True,
        },
        open=False,  # We open explicitly so we can retry on failure
    )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{attempt}/{max_retries}] Opening async connection pool...")
            await pool.open()

            # Verify Postgres is alive and pgvector is enabled
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
                result = await conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                vector_ext = await result.fetchone()
                
                if not vector_ext:
                    raise RuntimeError("pgvector extension is missing. Check postgres/schema.sql.")
                print(f"✅ pgvector extension confirmed: {vector_ext['extname']}")

            # Initialize LangGraph checkpoint tables
            print("Setting up AsyncPostgresSaver tables...")
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.asetup()

            print("✅ Async memory persistence initialized successfully.")
            await pool.close()
            return

        except Exception as exc:
            print(f"❌ Attempt {attempt} failed: {exc}")
            if attempt == max_retries:
                await pool.close()
                raise RuntimeError(
                    f"Database initialization failed after {max_retries} attempts. "
                    "Is the 'db' container healthy?"
                ) from exc

            # Exponential backoff: 2s, 3s, 4.5s, 6.75s...
            wait = base_delay * (1.5 ** (attempt - 1))
            print(f"⏳ Retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)


if __name__ == "__main__":
    asyncio.run(initialize_memory())
