"""split a transaction into multiple child transactions (multi-category split)

Revision ID: 066
Revises: 065
Create Date: 2026-07-13

Adds `transactions.parent_transaction_id`: a self-referential FK used when a
single bank transaction mixes two or more economic natures (e.g. a credit
that is partly a loan-principal refund and partly interest) and the user
splits it into N manual child transactions, each with its own amount and
category. The original row is kept (and flagged `is_ignored=True`) so history
isn't lost, while the children carry the split amounts that actually count
for reports. Nullable, ON DELETE SET NULL: deleting a parent (should be
blocked by the service layer while children exist) never orphans a dangling
reference on children that do somehow survive.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "066"
down_revision: Union[str, None] = "065"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "transactions",
        sa.Column(
            "parent_transaction_id",
            UUID(as_uuid=True),
            sa.ForeignKey("transactions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_transactions_parent_transaction_id",
        "transactions",
        ["parent_transaction_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_transactions_parent_transaction_id",
        table_name="transactions",
    )
    op.drop_column("transactions", "parent_transaction_id")
