"""Alembic environment. The database URL comes from DATABASE_URL (env or .env), never alembic.ini."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from yla.config import Secrets
from yla.db.models import Base
from yla.db.session import connect_args, normalize_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# A URL set programmatically (e.g. by the test suite) wins over the environment.
if not config.get_main_option("sqlalchemy.url"):
    database_url = Secrets().database_url
    if database_url is None:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    config.set_main_option("sqlalchemy.url", normalize_url(database_url.get_secret_value()))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args(config.get_main_option("sqlalchemy.url") or ""),
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
