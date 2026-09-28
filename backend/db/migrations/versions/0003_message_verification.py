"""persist message verification details"""

import sqlalchemy as sa
from alembic import op

revision = "0003_message_verification"
down_revision = "0002_google_subject"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add verification details needed to restore chat history."""
    op.add_column(
        "messages", sa.Column("verification_verified", sa.Boolean(), nullable=True)
    )
    op.add_column(
        "messages", sa.Column("verification_score", sa.Float(), nullable=True)
    )
    op.add_column(
        "messages", sa.Column("verification_explanation", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    """Remove persisted verification details."""
    op.drop_column("messages", "verification_explanation")
    op.drop_column("messages", "verification_score")
    op.drop_column("messages", "verification_verified")
