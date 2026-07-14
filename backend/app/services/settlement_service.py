import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.account import Account
from app.models.group import Group, GroupMember
from app.models.group_settlement import GroupSettlement
from app.models.transaction import Transaction
from app.schemas.group_settlement import (
    GroupSettlementCreate,
    GroupSettlementUpdate,
)


async def _ensure_group_visible(
    session: AsyncSession,
    group_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Optional[Group]:
    """Visible when the group lives in the current workspace OR the
    caller is linked as a cross-workspace member — for read endpoints."""
    from app.services.group_service import get_group_visible

    return await get_group_visible(session, group_id, workspace_id, user_id)


async def _user_member_id(
    session: AsyncSession, group_id: uuid.UUID, user_id: uuid.UUID
) -> Optional[uuid.UUID]:
    """If the user is a linked member of this group, return that member
    id. Owners may not have a linked member (they can still act via
    the owner check), so this can return None for them."""
    result = await session.execute(
        select(GroupMember.id).where(
            GroupMember.group_id == group_id,
            GroupMember.linked_user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def _can_settle_from(
    session: AsyncSession,
    group: Group,
    user_id: uuid.UUID,
    from_member_id: uuid.UUID,
) -> bool:
    """Permission check for creating/editing a settlement:
    - Group owner can do anything.
    - Linked member can only act when they are the `from_member`
      (i.e., they're recording a payment they themselves made)."""
    if group.user_id == user_id:
        return True
    linked = await _user_member_id(session, group.id, user_id)
    return linked is not None and linked == from_member_id


async def _create_payment_transaction(
    session: AsyncSession,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    account_id: uuid.UUID,
    amount,
    currency: str,
    when,
    description: str,
) -> Transaction:
    """Create a debit transaction on the user's account representing a
    settlement payment. Validates that the account belongs to the
    current workspace."""
    account_result = await session.execute(
        select(Account).where(
            Account.id == account_id,
            Account.workspace_id == workspace_id,
        )
    )
    account = account_result.scalar_one_or_none()
    if account is None:
        raise ValueError("Account not found")

    tx = Transaction(
        id=uuid.uuid4(),
        user_id=user_id,
        workspace_id=workspace_id,
        account_id=account.id,
        description=description,
        amount=amount,
        currency=currency,
        date=when,
        type="debit",
        # `settlement` is a special source that excludes the row from
        # spending reports — the underlying expense is already counted
        # via the share that produced the debt; this is the payback,
        # not a new expense.
        source="settlement",
        created_at=datetime.now(timezone.utc),
    )
    session.add(tx)
    await session.flush()
    # Stamp primary-currency amount so dashboard / report aggregations
    # that prefer amount_primary include this row.
    from app.services.fx_rate_service import stamp_primary_amount

    await stamp_primary_amount(session, user_id, tx)
    return tx


def _user_workspace_ids(user_id: uuid.UUID):
    """Subquery of every workspace the given user belongs to. Shared by
    the receiver-account lookup and the receiver-transaction validation —
    both need to reach across workspaces because the receiver may sit in
    a different workspace than the one the settlement was recorded in."""
    from app.models.workspace import WorkspaceMember

    return select(WorkspaceMember.workspace_id).where(
        WorkspaceMember.user_id == user_id
    )


async def _pick_default_account_for_user(
    session: AsyncSession, user_id: uuid.UUID
) -> Optional[Account]:
    """Return the user's first non-archived checking/savings account
    across any workspace they belong to. Used as the auto-target for
    receiver-side settlement credits — the receiver may sit in a
    different workspace than where the settlement was recorded."""
    result = await session.execute(
        select(Account)
        .where(
            Account.workspace_id.in_(_user_workspace_ids(user_id)),
            Account.is_closed.is_(False),
            Account.type.in_(("checking", "savings")),
        )
        .order_by(Account.name)
    )
    return result.scalars().first()


async def _create_receiver_credit(
    session: AsyncSession,
    receiver_user_id: uuid.UUID,
    amount,
    currency: str,
    when,
    description: str,
) -> Optional[Transaction]:
    """Mirror a settlement credit on the receiver's side.

    Picks the receiver's first checking/savings account from any
    workspace they belong to and stamps the credit there. Returns None
    silently when the receiver has no suitable account — they can do
    it manually or the next time they reconcile from their bank sync.
    """
    account = await _pick_default_account_for_user(session, receiver_user_id)
    if account is None:
        return None
    tx = Transaction(
        id=uuid.uuid4(),
        user_id=receiver_user_id,
        workspace_id=account.workspace_id,
        account_id=account.id,
        description=description,
        amount=amount,
        currency=currency,
        date=when,
        type="credit",
        # Settlement credits ARE counted in P/L (income), unlike the
        # payer's debit which is excluded. See counts_as_pnl().
        source="settlement",
        created_at=datetime.now(timezone.utc),
    )
    session.add(tx)
    await session.flush()
    from app.services.fx_rate_service import stamp_primary_amount

    await stamp_primary_amount(session, receiver_user_id, tx)
    return tx


async def _validate_members_in_group(
    session: AsyncSession, group_id: uuid.UUID, member_ids: list[uuid.UUID]
) -> None:
    result = await session.execute(
        select(GroupMember.id).where(
            GroupMember.group_id == group_id, GroupMember.id.in_(member_ids)
        )
    )
    found = {row[0] for row in result.all()}
    if found != set(member_ids):
        raise ValueError("Settlement members must belong to the group")


async def _validate_transaction(
    session: AsyncSession,
    transaction_id: Optional[uuid.UUID],
    workspace_id: uuid.UUID,
) -> None:
    if transaction_id is None:
        return
    result = await session.execute(
        select(Transaction).where(
            Transaction.id == transaction_id,
            Transaction.workspace_id == workspace_id,
        )
    )
    if result.scalar_one_or_none() is None:
        raise ValueError("Linked transaction not found")


async def _validate_receiver_transaction(
    session: AsyncSession,
    receiver_transaction_id: Optional[uuid.UUID],
    receiver_user_id: Optional[uuid.UUID],
) -> None:
    """Validate a receiver-side transaction link. Unlike the payer's
    `_validate_transaction`, this checks the transaction against every
    workspace the *receiver* belongs to, not the caller's workspace — the
    receiver may sit in a different workspace (same reasoning as
    `_pick_default_account_for_user`).

    Note: a linked existing transaction keeps its original `source` (e.g.
    "sync"), so it won't be excluded from P&L the way an auto-created
    settlement transaction is (see `counts_as_pnl`/`counts_as_user_pnl` in
    _query_filters.py, which key off `source == "settlement"`). This is
    the same known gap that already exists for the payer's `transaction_id`
    — accepted here rather than fixed, since correcting it means reworking
    report queries.
    """
    if receiver_transaction_id is None:
        return
    if receiver_user_id is None:
        raise ValueError(
            "Cannot link a receiver transaction: receiver has no resolvable Securo user"
        )
    result = await session.execute(
        select(Transaction).where(
            Transaction.id == receiver_transaction_id,
            Transaction.workspace_id.in_(_user_workspace_ids(receiver_user_id)),
        )
    )
    if result.scalar_one_or_none() is None:
        raise ValueError("Linked transaction not found")


async def _assert_transaction_available(
    session: AsyncSession,
    transaction_id: Optional[uuid.UUID],
    exclude_settlement_id: Optional[uuid.UUID] = None,
) -> None:
    """A transaction can back at most one settlement leg (either as the
    payer's `transaction_id` or the receiver's `receiver_transaction_id`
    of *any* settlement, including the same one) — otherwise the same
    real money movement would silently offset multiple debts, e.g. when
    recording several installment repayments and accidentally picking
    the same incoming transaction twice."""
    if transaction_id is None:
        return
    query = select(GroupSettlement.id).where(
        or_(
            GroupSettlement.transaction_id == transaction_id,
            GroupSettlement.receiver_transaction_id == transaction_id,
        )
    )
    if exclude_settlement_id is not None:
        query = query.where(GroupSettlement.id != exclude_settlement_id)
    result = await session.execute(query)
    if result.scalar_one_or_none() is not None:
        raise ValueError(
            "This transaction is already linked to another settlement"
        )


async def list_settlements(
    session: AsyncSession,
    group_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Optional[list[GroupSettlement]]:
    if not await _ensure_group_visible(session, group_id, workspace_id, user_id):
        return None
    result = await session.execute(
        select(GroupSettlement)
        .where(GroupSettlement.group_id == group_id)
        .order_by(GroupSettlement.date.desc(), GroupSettlement.created_at.desc())
    )
    return list(result.scalars().all())


async def create_settlement(
    session: AsyncSession,
    group_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    data: GroupSettlementCreate,
) -> Optional[GroupSettlement]:
    group = await _ensure_group_visible(session, group_id, workspace_id, user_id)
    if not group:
        return None

    if not await _can_settle_from(session, group, user_id, data.from_member_id):
        # Linked members may only record payments they themselves made.
        raise PermissionError(
            "You can only record settlements where you are the payer"
        )

    await _validate_members_in_group(
        session, group_id, [data.from_member_id, data.to_member_id]
    )
    await _validate_transaction(session, data.transaction_id, workspace_id)
    await _assert_transaction_available(session, data.transaction_id)

    payload = data.model_dump()
    account_id = payload.pop("account_id", None)
    description = payload.pop("description", None)
    skip_receiver_transaction = payload.pop("skip_receiver_transaction", False)
    receiver_transaction_id = payload.get("receiver_transaction_id")

    if receiver_transaction_id is not None and skip_receiver_transaction:
        raise ValueError(
            "Pass either receiver_transaction_id (to link an existing "
            "transaction) or skip_receiver_transaction (to skip it), not both"
        )
    if receiver_transaction_id is not None and receiver_transaction_id == data.transaction_id:
        raise ValueError(
            "A settlement cannot use the same transaction for both sides"
        )

    # Resolve member metadata once — we need names for descriptions
    # and the to_member's linked_user_id for the receiver-side credit.
    members_q = await session.execute(
        select(
            GroupMember.id,
            GroupMember.name,
            GroupMember.linked_user_id,
            GroupMember.is_self,
        ).where(GroupMember.id.in_([data.from_member_id, data.to_member_id]))
    )
    member_meta = {row.id: row for row in members_q.all()}
    from_name = member_meta[data.from_member_id].name if data.from_member_id in member_meta else "—"
    to_meta = member_meta.get(data.to_member_id)
    to_name = to_meta.name if to_meta else "—"
    # Resolve the receiver's Securo user id. linked_user_id wins; fall
    # back to group.user_id when the receiver is the owner's
    # self-member (owners often don't bother linking themselves).
    receiver_user_id = None
    if to_meta is not None:
        receiver_user_id = to_meta.linked_user_id
        if receiver_user_id is None and to_meta.is_self:
            receiver_user_id = group.user_id

    # Optional integration with the real account ledger: create a debit
    # transaction on the payer's account and link it via transaction_id.
    if account_id is not None:
        if payload.get("transaction_id") is not None:
            raise ValueError(
                "Pass either account_id (to create a transaction) or "
                "transaction_id (to link an existing one), not both"
            )
        auto_desc = description or f"Acerto · {group.name} · {to_name}"
        tx = await _create_payment_transaction(
            session,
            user_id,
            workspace_id,
            account_id,
            data.amount,
            data.currency,
            data.date,
            auto_desc,
        )
        payload["transaction_id"] = tx.id

    # Receiver-side: either link an existing credit transaction, skip it
    # entirely, or (default, unchanged) auto-create a mirror credit when
    # the receiver maps to a Securo user with a checking/savings account.
    if receiver_transaction_id is not None:
        await _validate_receiver_transaction(
            session, receiver_transaction_id, receiver_user_id
        )
        await _assert_transaction_available(session, receiver_transaction_id)
        receiver_tx_id = receiver_transaction_id
    elif skip_receiver_transaction:
        receiver_tx_id = None
    elif receiver_user_id is not None:
        receiver_desc = description or f"Acerto · {group.name} · {from_name}"
        receiver_tx = await _create_receiver_credit(
            session,
            receiver_user_id,
            data.amount,
            data.currency,
            data.date,
            receiver_desc,
        )
        receiver_tx_id = receiver_tx.id if receiver_tx is not None else None
    else:
        receiver_tx_id = None
    payload["receiver_transaction_id"] = receiver_tx_id

    settlement = GroupSettlement(
        group_id=group_id, workspace_id=workspace_id, **payload
    )
    session.add(settlement)
    await session.commit()
    await session.refresh(settlement)
    return settlement


async def update_settlement(
    session: AsyncSession,
    group_id: uuid.UUID,
    settlement_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    data: GroupSettlementUpdate,
) -> Optional[GroupSettlement]:
    group = await _ensure_group_visible(session, group_id, workspace_id, user_id)
    if not group:
        return None

    result = await session.execute(
        select(GroupSettlement).where(
            GroupSettlement.id == settlement_id,
            GroupSettlement.group_id == group_id,
        )
    )
    settlement = result.scalar_one_or_none()
    if not settlement:
        return None

    # Caller must currently own the settlement (linked member of the
    # original from_member, or the group owner).
    if not await _can_settle_from(session, group, user_id, settlement.from_member_id):
        raise PermissionError("You can only edit settlements you created")

    update_data = data.model_dump(exclude_unset=True)

    new_from = update_data.get("from_member_id", settlement.from_member_id)
    new_to = update_data.get("to_member_id", settlement.to_member_id)
    if new_from == new_to:
        raise ValueError("from_member_id and to_member_id must differ")

    member_check: list[uuid.UUID] = []
    if "from_member_id" in update_data:
        member_check.append(update_data["from_member_id"])
    if "to_member_id" in update_data:
        member_check.append(update_data["to_member_id"])
    if member_check:
        await _validate_members_in_group(session, group_id, member_check)

    if "transaction_id" in update_data:
        await _validate_transaction(session, update_data["transaction_id"], workspace_id)
        await _assert_transaction_available(
            session, update_data["transaction_id"], exclude_settlement_id=settlement.id
        )

    if "receiver_transaction_id" in update_data:
        to_meta_result = await session.execute(
            select(GroupMember.linked_user_id, GroupMember.is_self).where(
                GroupMember.id == new_to
            )
        )
        to_meta = to_meta_result.one_or_none()
        receiver_user_id = None
        if to_meta is not None:
            receiver_user_id = to_meta.linked_user_id
            if receiver_user_id is None and to_meta.is_self:
                receiver_user_id = group.user_id
        await _validate_receiver_transaction(
            session, update_data["receiver_transaction_id"], receiver_user_id
        )
        await _assert_transaction_available(
            session, update_data["receiver_transaction_id"], exclude_settlement_id=settlement.id
        )

    for key, value in update_data.items():
        setattr(settlement, key, value)

    await session.commit()
    await session.refresh(settlement)
    return settlement


async def delete_settlement(
    session: AsyncSession,
    group_id: uuid.UUID,
    settlement_id: uuid.UUID,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
) -> bool:
    group = await _ensure_group_visible(session, group_id, workspace_id, user_id)
    if not group:
        return False
    result = await session.execute(
        select(GroupSettlement).where(
            GroupSettlement.id == settlement_id,
            GroupSettlement.group_id == group_id,
        )
    )
    settlement = result.scalar_one_or_none()
    if not settlement:
        return False
    if not await _can_settle_from(session, group, user_id, settlement.from_member_id):
        raise PermissionError("You can only delete settlements you created")
    await session.delete(settlement)
    await session.commit()
    return True
