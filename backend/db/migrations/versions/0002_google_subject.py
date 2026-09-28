"""add Google identity subject"""

import sqlalchemy as sa
from alembic import op

revision = "0002_google_subject"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the optional Google identity mapping."""
    op.add_column(
        "users", sa.Column("google_subject", sa.String(length=255), nullable=True)
    )
    op.create_unique_constraint("uq_users_google_subject", "users", ["google_subject"])


def downgrade() -> None:
    """Remove the Google identity mapping."""
    op.drop_constraint("uq_users_google_subject", "users", type_="unique")
    op.drop_column("users", "google_subject")
