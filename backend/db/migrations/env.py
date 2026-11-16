"""Alembic environment configured from environment variables."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.config import database_url
from backend.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
config.set_main_option("sqlalchemy.url", database_url())
target_metadata = Base.metadata


def include_object(object_, name, type_, reflected, compare_to):  # type: ignore[no-untyped-def]
    """Exclude legacy vector-store tables from Alembic autogeneration."""
    return not (type_ == "table" and name in {"data_vectors", "aoa_embeddings"})


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the configured database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
