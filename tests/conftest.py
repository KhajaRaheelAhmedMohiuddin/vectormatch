import pytest


@pytest.fixture(scope="session", autouse=True)
def _drop_pgvector_test_tables():
    """Remove the throwaway pgvector tables created by the contract tests."""
    yield
    try:
        import psycopg

        from vectormatch.config import settings

        with psycopg.connect(settings.pg_dsn, autocommit=True, connect_timeout=2) as c:
            for (t,) in c.execute("SELECT tablename FROM pg_tables WHERE tablename LIKE 'vm_test_%'").fetchall():
                c.execute(f'DROP TABLE IF EXISTS "{t}"')
    except Exception:
        pass
