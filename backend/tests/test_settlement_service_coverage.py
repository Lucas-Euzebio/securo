"""Coverage-focused tests for settlement_service.

Targets _validate_transaction (valid + missing), list_settlements,
account_id + transaction_id conflict, linking an existing transaction,
update_settlement (not found / permission / member+transaction revalidation),
and delete_settlement (success / not found / permission).
"""
import uuid
from datetime import date
from decimal import Decimal

import bcrypt as _bcrypt
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.account import Account
from app.models.transaction import Transaction
from app.models.user import User
from app.schemas.group import GroupCreate, GroupMemberCreate
from app.schemas.group_settlement import (
    GroupSettlementCreate,
    GroupSettlementUpdate,
)
from app.services import group_service, settlement_service, workspace_service


async def _setup_group(session, user_id, workspace_id):
    group = await group_service.create_group(
        session, workspace_id, user_id, GroupCreate(name=f"S-{uuid.uuid4().hex[:6]}")
    )
    a = await group_service.create_member(
        session, group.id, workspace_id, GroupMemberCreate(name="Alice", is_self=True)
    )
    b = await group_service.create_member(
        session, group.id, workspace_id, GroupMemberCreate(name="Bob")
    )
    c = await group_service.create_member(
        session, group.id, workspace_id, GroupMemberCreate(name="Carol")
    )
    return group, a, b, c


async def _make_account(session, user_id, workspace_id):
    account = Account(
        id=uuid.uuid4(),
        user_id=user_id,
        workspace_id=workspace_id,
        name=f"Acc-{uuid.uuid4().hex[:4]}",
        type="checking",
        balance=Decimal("0"),
        currency="USD",
    )
    session.add(account)
    await session.flush()
    return account


async def _make_transaction(session, user_id, workspace_id, account_id, type="debit"):
    tx = Transaction(
        id=uuid.uuid4(),
        user_id=user_id,
        workspace_id=workspace_id,
        account_id=account_id,
        description="Existing",
        amount=Decimal("5.00"),
        currency="USD",
        date=date.today(),
        type=type,
        source="manual",
    )
    session.add(tx)
    await session.flush()
    return tx


@pytest.mark.asyncio
async def test_list_settlements(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("10.00"), currency="USD", date=date.today(),
        ),
    )
    listed = await settlement_service.list_settlements(
        session, group.id, test_workspace.id, test_user.id
    )
    assert listed is not None
    assert len(listed) == 1


@pytest.mark.asyncio
async def test_list_settlements_group_not_visible(session: AsyncSession, test_user, test_workspace):
    result = await settlement_service.list_settlements(
        session, uuid.uuid4(), test_workspace.id, test_user.id
    )
    assert result is None


@pytest.mark.asyncio
async def test_create_settlement_links_existing_transaction(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id)

    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("5.00"), currency="USD", date=date.today(),
            transaction_id=tx.id,
        ),
    )
    assert s is not None
    assert s.transaction_id == tx.id


@pytest.mark.asyncio
async def test_create_settlement_invalid_transaction(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    with pytest.raises(ValueError, match="Linked transaction not found"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=a.id, to_member_id=b.id,
                amount=Decimal("5.00"), currency="USD", date=date.today(),
                transaction_id=uuid.uuid4(),
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_account_and_transaction_conflict(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id)

    with pytest.raises(ValueError, match="either account_id"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=a.id, to_member_id=b.id,
                amount=Decimal("5.00"), currency="USD", date=date.today(),
                account_id=account.id, transaction_id=tx.id,
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_bad_account(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    with pytest.raises(ValueError, match="Account not found"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=a.id, to_member_id=b.id,
                amount=Decimal("5.00"), currency="USD", date=date.today(),
                account_id=uuid.uuid4(),
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_with_account_creates_payment_tx(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    account = await _make_account(session, test_user.id, test_workspace.id)

    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("25.00"), currency="USD", date=date.today(),
            account_id=account.id, description="Custom desc",
        ),
    )
    assert s is not None
    assert s.transaction_id is not None
    tx = await session.get(Transaction, s.transaction_id)
    assert tx is not None
    assert tx.type == "debit"
    assert tx.source == "settlement"
    assert tx.amount == Decimal("25.00")
    assert tx.description == "Custom desc"


@pytest.mark.asyncio
async def test_create_settlement_receiver_self_member_owner_fallback(session: AsyncSession, test_user, test_workspace):
    """to_member is the owner's self-member with no linked user -> receiver
    credit falls back to group.user_id and lands on the owner's account."""
    owner_account = await _make_account(session, test_user.id, test_workspace.id)
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="SelfRx")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )
    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=friend.id, to_member_id=owner_self.id,
            amount=Decimal("8.00"), currency="USD", date=date.today(),
        ),
    )
    assert s is not None
    assert s.receiver_transaction_id is not None
    rx = await session.get(Transaction, s.receiver_transaction_id)
    assert rx.account_id == owner_account.id
    assert rx.type == "credit"


@pytest.mark.asyncio
async def test_create_settlement_group_not_visible(session: AsyncSession, test_user, test_workspace):
    """A different workspace can't see the group -> create returns None."""
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    result = await settlement_service.create_settlement(
        session, group.id, uuid.uuid4(), test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("1.00"), currency="USD", date=date.today(),
        ),
    )
    assert result is None


@pytest.mark.asyncio
async def test_update_settlement_change_only_from_member(session: AsyncSession, test_user, test_workspace):
    """Changing only from_member_id triggers the single-member validation
    branch (covers the member_check append for from)."""
    group, a, b, c = await _setup_group(session, test_user.id, test_workspace.id)
    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("10.00"), currency="USD", date=date.today(),
        ),
    )
    updated = await settlement_service.update_settlement(
        session, group.id, s.id, test_workspace.id, test_user.id,
        GroupSettlementUpdate(from_member_id=c.id),
    )
    assert updated is not None
    assert updated.from_member_id == c.id


@pytest.mark.asyncio
async def test_update_settlement_not_found(session: AsyncSession, test_user, test_workspace):
    group, _a, _b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    result = await settlement_service.update_settlement(
        session, group.id, uuid.uuid4(), test_workspace.id, test_user.id,
        GroupSettlementUpdate(amount=Decimal("1.00")),
    )
    assert result is None


@pytest.mark.asyncio
async def test_update_settlement_group_not_visible(session: AsyncSession, test_user, test_workspace):
    result = await settlement_service.update_settlement(
        session, uuid.uuid4(), uuid.uuid4(), test_workspace.id, test_user.id,
        GroupSettlementUpdate(amount=Decimal("1.00")),
    )
    assert result is None


@pytest.mark.asyncio
async def test_update_settlement_revalidates_members_and_transaction(session: AsyncSession, test_user, test_workspace):
    group, a, b, c = await _setup_group(session, test_user.id, test_workspace.id)
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id)

    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("10.00"), currency="USD", date=date.today(),
        ),
    )
    # Change to_member to Carol and link the transaction.
    updated = await settlement_service.update_settlement(
        session, group.id, s.id, test_workspace.id, test_user.id,
        GroupSettlementUpdate(to_member_id=c.id, transaction_id=tx.id),
    )
    assert updated is not None
    assert updated.to_member_id == c.id
    assert updated.transaction_id == tx.id


@pytest.mark.asyncio
async def test_update_settlement_invalid_member(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("10.00"), currency="USD", date=date.today(),
        ),
    )
    with pytest.raises(ValueError, match="must belong to the group"):
        await settlement_service.update_settlement(
            session, group.id, s.id, test_workspace.id, test_user.id,
            GroupSettlementUpdate(to_member_id=uuid.uuid4()),
        )


@pytest.mark.asyncio
async def test_delete_settlement(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("10.00"), currency="USD", date=date.today(),
        ),
    )
    assert await settlement_service.delete_settlement(
        session, group.id, s.id, test_workspace.id, test_user.id
    ) is True
    # Now gone.
    assert await settlement_service.delete_settlement(
        session, group.id, s.id, test_workspace.id, test_user.id
    ) is False


@pytest.mark.asyncio
async def test_delete_settlement_group_not_visible(session: AsyncSession, test_user, test_workspace):
    assert await settlement_service.delete_settlement(
        session, uuid.uuid4(), uuid.uuid4(), test_workspace.id, test_user.id
    ) is False


@pytest.mark.asyncio
async def test_settlement_permission_for_linked_member(session: AsyncSession, test_user, test_workspace):
    """A linked non-owner member may only settle when they are the payer."""
    hashed = _bcrypt.hashpw(b"x", _bcrypt.gensalt()).decode()
    payer_user = User(
        id=uuid.uuid4(),
        email="payer@example.com",
        hashed_password=hashed,
        is_active=True,
        is_verified=True,
        preferences={"currency_display": "USD"},
    )
    session.add(payer_user)
    await session.flush()
    payer_ws = await workspace_service.create_personal_workspace_for_user(
        session, payer_user, commit=True
    )

    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="Perm")
    )
    me = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    payer = await group_service.create_member(
        session, group.id, test_workspace.id,
        GroupMemberCreate(name="Payer", linked_user_id=payer_user.id),
    )

    # The linked payer can record a payment where they are the from_member.
    s = await settlement_service.create_settlement(
        session, group.id, payer_ws.id, payer_user.id,
        GroupSettlementCreate(
            from_member_id=payer.id, to_member_id=me.id,
            amount=Decimal("3.00"), currency="USD", date=date.today(),
        ),
    )
    assert s is not None

    # But NOT one where someone else is the from_member.
    with pytest.raises(PermissionError, match="you are the payer"):
        await settlement_service.create_settlement(
            session, group.id, payer_ws.id, payer_user.id,
            GroupSettlementCreate(
                from_member_id=me.id, to_member_id=payer.id,
                amount=Decimal("3.00"), currency="USD", date=date.today(),
            ),
        )


@pytest.mark.asyncio
async def test_update_settlement_permission_denied(session: AsyncSession, test_user, test_workspace):
    """A linked member can't edit a settlement they don't own."""
    hashed = _bcrypt.hashpw(b"x", _bcrypt.gensalt()).decode()
    intruder = User(
        id=uuid.uuid4(),
        email="intruder@example.com",
        hashed_password=hashed,
        is_active=True,
        is_verified=True,
        preferences={"currency_display": "USD"},
    )
    session.add(intruder)
    await session.flush()
    intruder_ws = await workspace_service.create_personal_workspace_for_user(
        session, intruder, commit=True
    )

    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="Owned")
    )
    a = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Owner", is_self=True)
    )
    b = await group_service.create_member(
        session, group.id, test_workspace.id,
        GroupMemberCreate(name="Intruder", linked_user_id=intruder.id),
    )
    # Owner creates a settlement from a -> b.
    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("10.00"), currency="USD", date=date.today(),
        ),
    )
    # Intruder (linked to b, not the from_member) can't edit it.
    with pytest.raises(PermissionError, match="settlements you created"):
        await settlement_service.update_settlement(
            session, group.id, s.id, intruder_ws.id, intruder.id,
            GroupSettlementUpdate(amount=Decimal("99.00")),
        )
    # Nor delete it.
    with pytest.raises(PermissionError, match="settlements you created"):
        await settlement_service.delete_settlement(
            session, group.id, s.id, intruder_ws.id, intruder.id
        )


# ── Receiver-side transaction linking ──────────────────────────────────


@pytest.mark.asyncio
async def test_create_settlement_links_existing_receiver_transaction(session: AsyncSession, test_user, test_workspace):
    """to_member is the owner's self-member; linking an existing credit
    transaction on the receiver side must not also auto-create one."""
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id, type="credit")
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="RxLink")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )

    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=friend.id, to_member_id=owner_self.id,
            amount=Decimal("5.00"), currency="USD", date=date.today(),
            receiver_transaction_id=tx.id,
        ),
    )
    assert s is not None
    assert s.receiver_transaction_id == tx.id
    # No synthetic credit was created — the account still has just the one.
    txs = (await session.execute(
        select(Transaction).where(Transaction.account_id == account.id)
    )).scalars().all()
    assert len(txs) == 1


@pytest.mark.asyncio
async def test_create_settlement_skip_receiver_transaction(session: AsyncSession, test_user, test_workspace):
    """skip_receiver_transaction=True must not auto-create the mirror
    credit even though to_member.is_self resolves to a receiver."""
    await _make_account(session, test_user.id, test_workspace.id)
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="RxSkip")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )

    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=friend.id, to_member_id=owner_self.id,
            amount=Decimal("5.00"), currency="USD", date=date.today(),
            skip_receiver_transaction=True,
        ),
    )
    assert s is not None
    assert s.receiver_transaction_id is None


@pytest.mark.asyncio
async def test_create_settlement_receiver_transaction_and_skip_conflict(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id, type="credit")

    with pytest.raises(ValueError, match="either receiver_transaction_id"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=a.id, to_member_id=b.id,
                amount=Decimal("5.00"), currency="USD", date=date.today(),
                receiver_transaction_id=tx.id, skip_receiver_transaction=True,
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_receiver_transaction_same_as_payer_transaction(session: AsyncSession, test_user, test_workspace):
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id)

    with pytest.raises(ValueError, match="same transaction for both sides"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=a.id, to_member_id=b.id,
                amount=Decimal("5.00"), currency="USD", date=date.today(),
                transaction_id=tx.id, receiver_transaction_id=tx.id,
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_invalid_receiver_transaction(session: AsyncSession, test_user, test_workspace):
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="RxInvalid")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )
    with pytest.raises(ValueError, match="Linked transaction not found"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=friend.id, to_member_id=owner_self.id,
                amount=Decimal("5.00"), currency="USD", date=date.today(),
                receiver_transaction_id=uuid.uuid4(),
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_receiver_transaction_no_resolvable_user(session: AsyncSession, test_user, test_workspace):
    """to_member is a plain, unlinked, non-self member — there's no
    Securo user to validate the receiver transaction against."""
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id, type="credit")

    with pytest.raises(ValueError, match="no resolvable Securo user"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=a.id, to_member_id=b.id,
                amount=Decimal("5.00"), currency="USD", date=date.today(),
                receiver_transaction_id=tx.id,
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_receiver_transaction_wrong_workspace(session: AsyncSession, test_user, test_workspace):
    """The transaction exists, but lives in a workspace the resolved
    receiver (the owner, via to_member.is_self) doesn't belong to."""
    hashed = _bcrypt.hashpw(b"x", _bcrypt.gensalt()).decode()
    stranger = User(
        id=uuid.uuid4(),
        email="stranger@example.com",
        hashed_password=hashed,
        is_active=True,
        is_verified=True,
        preferences={"currency_display": "USD"},
    )
    session.add(stranger)
    await session.flush()
    stranger_ws = await workspace_service.create_personal_workspace_for_user(
        session, stranger, commit=True
    )
    stranger_account = await _make_account(session, stranger.id, stranger_ws.id)
    foreign_tx = await _make_transaction(session, stranger.id, stranger_ws.id, stranger_account.id, type="credit")

    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="RxWrongWs")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )

    with pytest.raises(ValueError, match="Linked transaction not found"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=friend.id, to_member_id=owner_self.id,
                amount=Decimal("5.00"), currency="USD", date=date.today(),
                receiver_transaction_id=foreign_tx.id,
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_receiver_transaction_not_excluded_from_pnl(session: AsyncSession, test_user, test_workspace):
    """Known gap, documented on purpose: a linked existing receiver
    transaction keeps its original `source`, so it is NOT excluded by
    counts_as_user_pnl() the way an auto-created settlement credit is."""
    from app.services._query_filters import counts_as_user_pnl

    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id, type="credit")
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="RxPnl")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )
    await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=friend.id, to_member_id=owner_self.id,
            amount=Decimal("5.00"), currency="USD", date=date.today(),
            receiver_transaction_id=tx.id,
        ),
    )
    assert tx.source == "manual"
    still_counted = (await session.execute(
        select(Transaction).where(Transaction.id == tx.id, counts_as_user_pnl())
    )).scalar_one_or_none()
    assert still_counted is not None


@pytest.mark.asyncio
async def test_update_settlement_revalidates_receiver_transaction(session: AsyncSession, test_user, test_workspace):
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="RxUpdate")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id, type="credit")

    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=friend.id, to_member_id=owner_self.id,
            amount=Decimal("5.00"), currency="USD", date=date.today(),
            skip_receiver_transaction=True,
        ),
    )
    updated = await settlement_service.update_settlement(
        session, group.id, s.id, test_workspace.id, test_user.id,
        GroupSettlementUpdate(receiver_transaction_id=tx.id),
    )
    assert updated is not None
    assert updated.receiver_transaction_id == tx.id


@pytest.mark.asyncio
async def test_update_settlement_invalid_receiver_transaction(session: AsyncSession, test_user, test_workspace):
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="RxUpdateBad")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )
    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=friend.id, to_member_id=owner_self.id,
            amount=Decimal("5.00"), currency="USD", date=date.today(),
            skip_receiver_transaction=True,
        ),
    )
    with pytest.raises(ValueError, match="Linked transaction not found"):
        await settlement_service.update_settlement(
            session, group.id, s.id, test_workspace.id, test_user.id,
            GroupSettlementUpdate(receiver_transaction_id=uuid.uuid4()),
        )


# ── A transaction can back at most one settlement leg ──────────────────
# Regression coverage for: recording installment repayments (3x) let the
# same real transaction get linked to two different settlements, double-
# offsetting the debt from a single money movement.


@pytest.mark.asyncio
async def test_create_settlement_transaction_already_linked_as_payer(session: AsyncSession, test_user, test_workspace):
    group, a, b, c = await _setup_group(session, test_user.id, test_workspace.id)
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id)

    await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("2.00"), currency="USD", date=date.today(),
            transaction_id=tx.id,
        ),
    )
    with pytest.raises(ValueError, match="already linked to another settlement"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=a.id, to_member_id=c.id,
                amount=Decimal("3.00"), currency="USD", date=date.today(),
                transaction_id=tx.id,
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_receiver_transaction_already_linked(session: AsyncSession, test_user, test_workspace):
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id, type="credit")
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="RxDup")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )

    await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=friend.id, to_member_id=owner_self.id,
            amount=Decimal("2.00"), currency="USD", date=date.today(),
            receiver_transaction_id=tx.id,
        ),
    )
    # Simulates recording a second installment but accidentally picking
    # the same already-linked incoming transaction again.
    with pytest.raises(ValueError, match="already linked to another settlement"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=friend.id, to_member_id=owner_self.id,
                amount=Decimal("3.00"), currency="USD", date=date.today(),
                receiver_transaction_id=tx.id,
            ),
        )


@pytest.mark.asyncio
async def test_create_settlement_transaction_already_linked_cross_side(session: AsyncSession, test_user, test_workspace):
    """A transaction linked as one settlement's payer leg cannot also be
    linked as another settlement's receiver leg."""
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id)
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id, GroupCreate(name="RxCross")
    )
    owner_self = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True)
    )
    friend = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Friend")
    )

    await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=owner_self.id, to_member_id=friend.id,
            amount=Decimal("2.00"), currency="USD", date=date.today(),
            transaction_id=tx.id,
        ),
    )
    with pytest.raises(ValueError, match="already linked to another settlement"):
        await settlement_service.create_settlement(
            session, group.id, test_workspace.id, test_user.id,
            GroupSettlementCreate(
                from_member_id=friend.id, to_member_id=owner_self.id,
                amount=Decimal("3.00"), currency="USD", date=date.today(),
                receiver_transaction_id=tx.id,
            ),
        )


@pytest.mark.asyncio
async def test_update_settlement_transaction_still_available_when_unchanged(session: AsyncSession, test_user, test_workspace):
    """Re-saving a settlement with the transaction it already owns must
    not trip the already-linked check against itself."""
    group, a, b, _c = await _setup_group(session, test_user.id, test_workspace.id)
    account = await _make_account(session, test_user.id, test_workspace.id)
    tx = await _make_transaction(session, test_user.id, test_workspace.id, account.id)

    s = await settlement_service.create_settlement(
        session, group.id, test_workspace.id, test_user.id,
        GroupSettlementCreate(
            from_member_id=a.id, to_member_id=b.id,
            amount=Decimal("2.00"), currency="USD", date=date.today(),
            transaction_id=tx.id,
        ),
    )
    updated = await settlement_service.update_settlement(
        session, group.id, s.id, test_workspace.id, test_user.id,
        GroupSettlementUpdate(transaction_id=tx.id, notes="unchanged tx, just adding a note"),
    )
    assert updated is not None
    assert updated.transaction_id == tx.id
