import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.account import Account
from app.models.category import Category
from app.models.credit_card_bill import CreditCardBill
from app.models.group import Group, GroupMember
from app.models.group_settlement import GroupSettlement
from app.models.rule import Rule
from app.models.transaction import Transaction
from app.models.transaction_split import TransactionSplit
from app.providers.base import TransactionData
from app.schemas.group import GroupCreate, GroupMemberCreate
from app.schemas.transaction import (
    TransactionCreate,
    TransactionSplitPartInput,
    TransactionUpdate,
    TransferCreate,
)
from app.services import group_service
from app.services.connection_service import _fuzzy_match_manual
from app.services.transfer_detection_service import detect_transfer_pairs
from app.services.transaction_service import (
    _apply_fx_override,
    bulk_add_tags,
    bulk_remove_tags,
    bulk_update_category,
    create_transaction,
    create_transfer,
    delete_transaction,
    revert_split_transaction,
    split_transaction_into_parts,
    toggle_ignore_transaction,
    get_transaction,
    get_transactions,
    update_transaction,
)


@pytest_asyncio.fixture
async def txn_account(session: AsyncSession, test_user) -> Account:
    account = Account(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="TxnAcc",
        type="checking",
        balance=Decimal("10000"),
        currency="BRL",
    )
    session.add(account)
    await session.commit()
    await session.refresh(account)
    return account


# ---------------------------------------------------------------------------
# create_transaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_transaction_manual(
    session: AsyncSession, test_user, test_workspace, test_categories, txn_account
):
    data = TransactionCreate(
        description="Lunch",
        amount=Decimal("35.00"),
        date=date(2025, 3, 10),
        type="debit",
        account_id=txn_account.id,
        category_id=test_categories[0].id,
    )
    txn = await create_transaction(session, test_workspace.id, test_user.id, data)

    assert txn.id is not None
    assert txn.description == "Lunch"
    assert txn.source == "manual"
    assert txn.category_id == test_categories[0].id


@pytest.mark.asyncio
async def test_create_transaction_honors_effective_bill_date(
    session: AsyncSession, test_user, test_workspace
):
    account = Account(
        id=uuid.uuid4(),
        user_id=test_user.id,
        workspace_id=test_workspace.id,
        name="Card",
        type="credit_card",
        balance=Decimal("0"),
        currency="BRL",
        statement_close_day=10,
        payment_due_day=20,
    )
    bill = CreditCardBill(
        id=uuid.uuid4(),
        user_id=test_user.id,
        workspace_id=test_workspace.id,
        account_id=account.id,
        external_id="bill-1",
        due_date=date(2025, 5, 20),
        total_amount=Decimal("100"),
        currency="BRL",
    )
    session.add_all([account, bill])
    await session.commit()

    txn = await create_transaction(
        session,
        test_workspace.id,
        test_user.id,
        TransactionCreate(
            description="Manual charge",
            amount=Decimal("50.00"),
            date=date(2025, 3, 5),
            type="debit",
            account_id=account.id,
            effective_bill_date=date(2025, 5, 20),
        ),
    )

    assert txn.effective_bill_date == date(2025, 5, 20)
    assert txn.effective_date == date(2025, 5, 20)
    assert txn.bill_id == bill.id


@pytest.mark.asyncio
async def test_create_transaction_applies_rules(
    session: AsyncSession, test_user, test_workspace, test_categories, txn_account
):
    # Create a rule
    rule = Rule(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="UBER auto",
        conditions_op="or",
        conditions=[{"field": "description", "op": "starts_with", "value": "UBER"}],
        actions=[{"op": "set_category", "value": str(test_categories[1].id)}],
        priority=10,
        is_active=True,
    )
    session.add(rule)
    await session.commit()

    # Create transaction without category — rule should apply
    data = TransactionCreate(
        description="UBER TRIP",
        amount=Decimal("25"),
        date=date(2025, 3, 10),
        type="debit",
        account_id=txn_account.id,
    )
    txn = await create_transaction(session, test_workspace.id, test_user.id, data)

    assert txn.category_id == test_categories[1].id


@pytest.mark.asyncio
async def test_create_transaction_with_category_skips_rules(
    session: AsyncSession, test_user, test_workspace, test_categories, txn_account
):
    rule = Rule(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="UBER skip",
        conditions_op="or",
        conditions=[{"field": "description", "op": "starts_with", "value": "UBER"}],
        actions=[{"op": "set_category", "value": str(test_categories[1].id)}],
        priority=10,
        is_active=True,
    )
    session.add(rule)
    await session.commit()

    # Explicitly provide a different category — rule should NOT override
    data = TransactionCreate(
        description="UBER TRIP",
        amount=Decimal("25"),
        date=date(2025, 3, 10),
        type="debit",
        account_id=txn_account.id,
        category_id=test_categories[0].id,
    )
    txn = await create_transaction(session, test_workspace.id, test_user.id, data)
    assert txn.category_id == test_categories[0].id


@pytest.mark.asyncio
async def test_create_transaction_invalid_account(session: AsyncSession, test_user, test_workspace):
    data = TransactionCreate(
        description="Orphan",
        amount=Decimal("10"),
        date=date(2025, 3, 10),
        type="debit",
        account_id=uuid.uuid4(),
    )
    with pytest.raises(ValueError, match="Account not found"):
        await create_transaction(session, test_workspace.id, test_user.id, data)


# ---------------------------------------------------------------------------
# get_transactions — pagination & filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_transactions_pagination(session: AsyncSession, test_user, test_workspace, txn_account):
    # Create 5 transactions
    for i in range(5):
        txn = Transaction(
            id=uuid.uuid4(),
            user_id=test_user.id,
            account_id=txn_account.id,
            description=f"Txn {i}",
            amount=Decimal("10"),
            date=date(2025, 3, i + 1),
            type="debit",
            source="manual",
            created_at=datetime.now(timezone.utc),
        )
        session.add(txn)
    await session.commit()

    page1, total, _ = await get_transactions(session, test_workspace.id, test_user.id, limit=2, page=1)
    assert total >= 5
    assert len(page1) == 2

    page2, _, _ = await get_transactions(session, test_workspace.id, test_user.id, limit=2, page=2)
    assert len(page2) == 2

    # No overlap
    p1_ids = {t.id for t in page1}
    p2_ids = {t.id for t in page2}
    assert p1_ids.isdisjoint(p2_ids)


@pytest.mark.asyncio
async def test_get_transactions_filter_by_account(session: AsyncSession, test_user, test_workspace, txn_account):
    other_account = Account(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="Other",
        type="savings",
        balance=Decimal("0"),
        currency="BRL",
    )
    session.add(other_account)
    await session.commit()

    txn1 = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="In main",
        amount=Decimal("10"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    txn2 = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=other_account.id,
        description="In other",
        amount=Decimal("20"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add_all([txn1, txn2])
    await session.commit()

    results, _, _ = await get_transactions(session, test_workspace.id, test_user.id, account_id=txn_account.id)
    descs = {t.description for t in results}
    assert "In main" in descs
    assert "In other" not in descs


@pytest.mark.asyncio
async def test_get_transactions_filter_by_category(
    session: AsyncSession, test_user, test_workspace, test_categories, txn_account
):
    txn1 = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        category_id=test_categories[0].id,
        description="Cat A",
        amount=Decimal("10"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    txn2 = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        category_id=test_categories[1].id,
        description="Cat B",
        amount=Decimal("20"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add_all([txn1, txn2])
    await session.commit()

    results, _, _ = await get_transactions(session, test_workspace.id, test_user.id, category_id=test_categories[0].id)
    descs = {t.description for t in results}
    assert "Cat A" in descs
    assert "Cat B" not in descs


@pytest.mark.asyncio
async def test_get_transactions_filter_by_date_range(session: AsyncSession, test_user, test_workspace, txn_account):
    txn_jan = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="Jan",
        amount=Decimal("10"),
        date=date(2025, 1, 15),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    txn_mar = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="Mar",
        amount=Decimal("10"),
        date=date(2025, 3, 15),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add_all([txn_jan, txn_mar])
    await session.commit()

    results, _, _ = await get_transactions(
        session,
        test_workspace.id, test_user.id,
        from_date=date(2025, 3, 1),
        to_date=date(2025, 3, 31),
    )
    descs = {t.description for t in results}
    assert "Mar" in descs
    assert "Jan" not in descs


@pytest.mark.asyncio
async def test_get_transactions_date_filter_respects_accounting_mode(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    # CC purchase on Mar 25 that bills on Apr 15 (effective_date shifted).
    cc_purchase = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="CC purchase",
        amount=Decimal("50"),
        date=date(2025, 3, 25),
        effective_date=date(2025, 4, 15),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    # Regular tx that sits on its own date in both modes.
    regular = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="Regular",
        amount=Decimal("20"),
        date=date(2025, 4, 5),
        effective_date=date(2025, 4, 5),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add_all([cc_purchase, regular])
    await session.commit()

    april_window = dict(from_date=date(2025, 4, 1), to_date=date(2025, 4, 30))
    march_window = dict(from_date=date(2025, 3, 1), to_date=date(2025, 3, 31))

    # Cash mode (default): the CC purchase lives in March, regular in April.
    cash_april, _, _ = await get_transactions(session, test_workspace.id, test_user.id, **april_window)
    assert {t.description for t in cash_april} == {"Regular"}

    cash_march, _, _ = await get_transactions(session, test_workspace.id, test_user.id, **march_window)
    assert {t.description for t in cash_march} == {"CC purchase"}

    # Accrual mode: both buckets shift to the bill-due month.
    accrual_april, _, _ = await get_transactions(
        session, test_workspace.id, test_user.id, accounting_mode="accrual", **april_window
    )
    assert {t.description for t in accrual_april} == {"CC purchase", "Regular"}

    accrual_march, _, _ = await get_transactions(
        session, test_workspace.id, test_user.id, accounting_mode="accrual", **march_window
    )
    assert {t.description for t in accrual_march} == set()


@pytest.mark.asyncio
async def test_get_transactions_filter_by_search(session: AsyncSession, test_user, test_workspace, txn_account):
    txn = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="NETFLIX SUBSCRIPTION",
        amount=Decimal("39.90"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add(txn)
    await session.commit()

    results, _, _ = await get_transactions(session, test_workspace.id, test_user.id, search="netflix")
    descs = {t.description for t in results}
    assert "NETFLIX SUBSCRIPTION" in descs


@pytest.mark.asyncio
async def test_get_transactions_filter_by_type(session: AsyncSession, test_user, test_workspace, txn_account):
    txn_debit = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="Expense",
        amount=Decimal("50"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    txn_credit = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="Income",
        amount=Decimal("1000"),
        date=date(2025, 3, 1),
        type="credit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add_all([txn_debit, txn_credit])
    await session.commit()

    results, _, _ = await get_transactions(session, test_workspace.id, test_user.id, txn_type="credit")
    types = {t.type for t in results}
    assert "credit" in types
    assert all(t.type == "credit" for t in results)


# ---------------------------------------------------------------------------
# get_transaction / update_transaction / delete_transaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_transaction_by_id(session: AsyncSession, test_user, test_workspace, txn_account):
    txn = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="Lookup",
        amount=Decimal("10"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add(txn)
    await session.commit()

    fetched = await get_transaction(session, txn.id, test_workspace.id)
    assert fetched is not None
    assert fetched.id == txn.id


@pytest.mark.asyncio
async def test_get_transaction_not_found(session: AsyncSession, test_user, test_workspace):
    result = await get_transaction(session, uuid.uuid4(), test_workspace.id)
    assert result is None


@pytest.mark.asyncio
async def test_update_transaction(session: AsyncSession, test_user, test_workspace, txn_account):
    txn = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="Old",
        amount=Decimal("10"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add(txn)
    await session.commit()

    updated = await update_transaction(
        session,
        txn.id,
        test_workspace.id, test_user.id,
        TransactionUpdate(description="New", amount=Decimal("99")),
    )
    assert updated is not None
    assert updated.description == "New"
    assert updated.amount == Decimal("99")


@pytest.mark.asyncio
async def test_update_transaction_not_found(session: AsyncSession, test_user, test_workspace):
    result = await update_transaction(
        session,
        uuid.uuid4(),
        test_workspace.id,
        test_user.id,
        TransactionUpdate(description="Ghost"),
    )
    assert result is None


@pytest.mark.asyncio
async def test_delete_transaction(session: AsyncSession, test_user, test_workspace, txn_account):
    txn = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="ToDelete",
        amount=Decimal("5"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add(txn)
    await session.commit()

    assert await delete_transaction(session, txn.id, test_workspace.id) is True
    assert await get_transaction(session, txn.id, test_workspace.id) is None


@pytest.mark.asyncio
async def test_delete_transaction_not_found(session: AsyncSession, test_user, test_workspace):
    assert await delete_transaction(session, uuid.uuid4(), test_workspace.id) is False


@pytest.mark.asyncio
async def test_toggle_ignore_transaction(session: AsyncSession, test_user, test_workspace, txn_account):
    txn = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="ToIgnore",
        amount=Decimal("10"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add(txn)
    await session.commit()

    # First toggle: false → true
    result = await toggle_ignore_transaction(session, txn.id, test_workspace.id)
    assert result is not None
    assert result.is_ignored is True

    # Second toggle: true → false
    result = await toggle_ignore_transaction(session, txn.id, test_workspace.id)
    assert result is not None
    assert result.is_ignored is False


@pytest.mark.asyncio
async def test_toggle_ignore_transaction_not_found(session: AsyncSession, test_user, test_workspace):
    assert await toggle_ignore_transaction(session, uuid.uuid4(), test_workspace.id) is None


# ---------------------------------------------------------------------------
# bulk_update_category
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulk_update_category(session: AsyncSession, test_user, test_workspace, test_categories, txn_account):
    txns = []
    for i in range(3):
        txn = Transaction(
            id=uuid.uuid4(),
            user_id=test_user.id,
            account_id=txn_account.id,
            description=f"Bulk {i}",
            amount=Decimal("10"),
            date=date(2025, 3, i + 1),
            type="debit",
            source="manual",
            created_at=datetime.now(timezone.utc),
        )
        session.add(txn)
        txns.append(txn)
    await session.commit()

    ids = [t.id for t in txns]
    count = await bulk_update_category(session, test_workspace.id, ids, test_categories[0].id)
    assert count == 3

    for txn in txns:
        await session.refresh(txn)
        assert txn.category_id == test_categories[0].id


@pytest.mark.asyncio
async def test_bulk_update_category_clear(
    session: AsyncSession, test_user, test_workspace, test_categories, txn_account
):
    txn = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=txn_account.id,
        description="ClearCat",
        amount=Decimal("10"),
        date=date(2025, 3, 1),
        type="debit",
        source="manual",
        category_id=test_categories[0].id,
        created_at=datetime.now(timezone.utc),
    )
    session.add(txn)
    await session.commit()

    count = await bulk_update_category(session, test_workspace.id, [txn.id], category_id=None)
    assert count == 1

    await session.refresh(txn)
    assert txn.category_id is None


# ---------------------------------------------------------------------------
# _apply_fx_override
# ---------------------------------------------------------------------------


def test_apply_fx_override_both():
    txn = Transaction(
        id=uuid.uuid4(), user_id=uuid.uuid4(), account_id=uuid.uuid4(),
        description="T", amount=Decimal("100"), date=date.today(),
        type="debit", source="manual",
    )
    _apply_fx_override(txn, 100, amount_primary=500.0, fx_rate_used=5.0)
    assert txn.amount_primary == Decimal("500.0")
    assert txn.fx_rate_used == Decimal("5.0")


def test_apply_fx_override_only_amount_primary():
    txn = Transaction(
        id=uuid.uuid4(), user_id=uuid.uuid4(), account_id=uuid.uuid4(),
        description="T", amount=Decimal("100"), date=date.today(),
        type="debit", source="manual",
    )
    _apply_fx_override(txn, 100, amount_primary=250.0)
    assert txn.amount_primary == Decimal("250.0")
    assert txn.fx_rate_used == Decimal("2.5")


def test_apply_fx_override_only_fx_rate():
    txn = Transaction(
        id=uuid.uuid4(), user_id=uuid.uuid4(), account_id=uuid.uuid4(),
        description="T", amount=Decimal("100"), date=date.today(),
        type="debit", source="manual",
    )
    _apply_fx_override(txn, 100, fx_rate_used=3.0)
    assert txn.fx_rate_used == Decimal("3.0")
    assert txn.amount_primary == Decimal("300.00")


def test_apply_fx_override_zero_amount():
    txn = Transaction(
        id=uuid.uuid4(), user_id=uuid.uuid4(), account_id=uuid.uuid4(),
        description="T", amount=Decimal("0"), date=date.today(),
        type="debit", source="manual",
    )
    _apply_fx_override(txn, 0, amount_primary=0.0)
    assert txn.amount_primary == Decimal("0.0")
    assert txn.fx_rate_used == Decimal("1")


# ---------------------------------------------------------------------------
# create_transaction — FX and category
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_transaction_with_fx_override(session: AsyncSession, test_user, test_workspace, txn_account):
    data = TransactionCreate(
        account_id=txn_account.id, description="USD Purchase",
        amount=Decimal("100"), date=date.today(), type="debit",
        amount_primary=Decimal("500"), fx_rate_used=Decimal("5"),
    )
    txn = await create_transaction(session, test_workspace.id, test_user.id, data)
    assert txn.amount_primary == Decimal("500")
    assert txn.fx_rate_used == Decimal("5")


# ---------------------------------------------------------------------------
# create_transfer
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def txn_account_usd(session: AsyncSession, test_user) -> Account:
    acct = Account(
        id=uuid.uuid4(), user_id=test_user.id, name="USD Acct",
        type="checking", balance=Decimal("5000"), currency="USD",
    )
    session.add(acct)
    await session.commit()
    await session.refresh(acct)
    return acct


@pytest.mark.asyncio
async def test_create_transfer_same_currency(session: AsyncSession, test_user, test_workspace, txn_account):
    acct2 = Account(
        id=uuid.uuid4(), user_id=test_user.id, name="Savings",
        type="savings", balance=Decimal("0"), currency="BRL",
    )
    session.add(acct2)
    await session.commit()

    data = TransferCreate(
        from_account_id=txn_account.id, to_account_id=acct2.id,
        description="Transfer", amount=Decimal("1000"), date=date.today(),
    )
    debit_tx, credit_tx = await create_transfer(session, test_workspace.id, test_user.id, data)
    assert debit_tx.type == "debit"
    assert credit_tx.type == "credit"
    assert debit_tx.transfer_pair_id == credit_tx.transfer_pair_id
    assert debit_tx.amount == credit_tx.amount


@pytest.mark.asyncio
async def test_create_transfer_same_account(session: AsyncSession, test_user, test_workspace, txn_account):
    data = TransferCreate(
        from_account_id=txn_account.id, to_account_id=txn_account.id,
        description="Self", amount=Decimal("100"), date=date.today(),
    )
    with pytest.raises(ValueError, match="same account"):
        await create_transfer(session, test_workspace.id, test_user.id, data)


@pytest.mark.asyncio
async def test_create_transfer_cross_currency(session: AsyncSession, test_user, test_workspace, txn_account, txn_account_usd):
    data = TransferCreate(
        from_account_id=txn_account.id, to_account_id=txn_account_usd.id,
        description="Cross-currency", amount=Decimal("500"), date=date.today(),
        fx_rate=Decimal("0.2"),
    )
    debit_tx, credit_tx = await create_transfer(session, test_workspace.id, test_user.id, data)
    assert debit_tx.currency == "BRL"
    assert credit_tx.currency == "USD"
    assert credit_tx.amount == Decimal("100.00")


@pytest.mark.asyncio
async def test_create_transfer_cross_currency_auto_fx(session: AsyncSession, test_user, test_workspace, txn_account, txn_account_usd):
    data = TransferCreate(
        from_account_id=txn_account.id, to_account_id=txn_account_usd.id,
        description="Auto FX", amount=Decimal("500"), date=date.today(),
    )
    debit_tx, credit_tx = await create_transfer(session, test_workspace.id, test_user.id, data)
    assert debit_tx.type == "debit"
    assert credit_tx.type == "credit"


@pytest.mark.asyncio
async def test_create_transfer_invalid_from_account(session: AsyncSession, test_user, test_workspace, txn_account):
    data = TransferCreate(
        from_account_id=uuid.uuid4(), to_account_id=txn_account.id,
        description="Bad from", amount=Decimal("100"), date=date.today(),
    )
    with pytest.raises(ValueError, match="Source account not found"):
        await create_transfer(session, test_workspace.id, test_user.id, data)


@pytest.mark.asyncio
async def test_create_transfer_invalid_to_account(session: AsyncSession, test_user, test_workspace, txn_account):
    data = TransferCreate(
        from_account_id=txn_account.id, to_account_id=uuid.uuid4(),
        description="Bad to", amount=Decimal("100"), date=date.today(),
    )
    with pytest.raises(ValueError, match="Destination account not found"):
        await create_transfer(session, test_workspace.id, test_user.id, data)


# ---------------------------------------------------------------------------
# get_transactions — additional filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_transactions_uncategorized(session: AsyncSession, test_user, test_workspace, txn_account, test_categories):
    await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
        account_id=txn_account.id, description="Uncategorized",
        amount=Decimal("50"), date=date.today(), type="debit",
    ))
    await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
        account_id=txn_account.id, description="Categorized",
        amount=Decimal("50"), date=date.today(), type="debit",
        category_id=test_categories[0].id,
    ))
    txns, _, _ = await get_transactions(session, test_workspace.id, test_user.id, uncategorized=True)
    descs = [t.description for t in txns]
    assert "Uncategorized" in descs
    assert "Categorized" not in descs


@pytest.mark.asyncio
async def test_get_transactions_date_filter(session: AsyncSession, test_user, test_workspace, txn_account):
    from datetime import timedelta
    today = date.today()
    yesterday = today - timedelta(days=1)
    await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
        account_id=txn_account.id, description="Today",
        amount=Decimal("10"), date=today, type="debit",
    ))
    await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
        account_id=txn_account.id, description="Yesterday",
        amount=Decimal("10"), date=yesterday, type="debit",
    ))
    txns, _, _ = await get_transactions(session, test_workspace.id, test_user.id, from_date=today, to_date=today)
    descs = [t.description for t in txns]
    assert "Today" in descs
    assert "Yesterday" not in descs


@pytest.mark.asyncio
async def test_get_transactions_exclude_transfers(session: AsyncSession, test_user, test_workspace, txn_account):
    acct2 = Account(
        id=uuid.uuid4(), user_id=test_user.id, name="Sav",
        type="savings", balance=Decimal("0"), currency="BRL",
    )
    session.add(acct2)
    await session.commit()

    await create_transfer(session, test_workspace.id, test_user.id, TransferCreate(
        from_account_id=txn_account.id, to_account_id=acct2.id,
        description="Xfer", amount=Decimal("100"), date=date.today(),
    ))
    await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
        account_id=txn_account.id, description="Regular",
        amount=Decimal("50"), date=date.today(), type="debit",
    ))
    txns, _, _ = await get_transactions(session, test_workspace.id, test_user.id, exclude_transfers=True)
    descs = [t.description for t in txns]
    assert "Regular" in descs


@pytest.mark.asyncio
async def test_get_transactions_skip_pagination(session: AsyncSession, test_user, test_workspace, txn_account):
    for i in range(5):
        await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
            account_id=txn_account.id, description=f"Txn{i}",
            amount=Decimal("10"), date=date.today(), type="debit",
        ))
    txns, total, _ = await get_transactions(session, test_workspace.id, test_user.id, skip_pagination=True, limit=2)
    assert len(txns) == total


# ---------------------------------------------------------------------------
# update_transaction — FX and cascade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_transaction_fx_override(session: AsyncSession, test_user, test_workspace, txn_account):
    txn = await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
        account_id=txn_account.id, description="FX Test",
        amount=Decimal("100"), date=date.today(), type="debit",
    ))
    data = TransactionUpdate(amount_primary=Decimal("500"), fx_rate_used=Decimal("5"))
    updated = await update_transaction(session, txn.id, test_workspace.id, test_user.id, data)
    assert updated.amount_primary == Decimal("500")


@pytest.mark.asyncio
async def test_update_transaction_restamp_on_amount_change(session: AsyncSession, test_user, test_workspace, txn_account):
    txn = await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
        account_id=txn_account.id, description="Restamp",
        amount=Decimal("100"), date=date.today(), type="debit",
    ))
    data = TransactionUpdate(amount=Decimal("200"))
    updated = await update_transaction(session, txn.id, test_workspace.id, test_user.id, data)
    assert updated.amount == Decimal("200")


@pytest.mark.asyncio
async def test_update_transfer_cascades(session: AsyncSession, test_user, test_workspace, txn_account):
    acct2 = Account(
        id=uuid.uuid4(), user_id=test_user.id, name="CascSav",
        type="savings", balance=Decimal("0"), currency="BRL",
    )
    session.add(acct2)
    await session.commit()

    debit_tx, credit_tx = await create_transfer(session, test_workspace.id, test_user.id, TransferCreate(
        from_account_id=txn_account.id, to_account_id=acct2.id,
        description="Cascade Xfer", amount=Decimal("200"), date=date.today(),
    ))
    data = TransactionUpdate(description="Updated Xfer")
    updated = await update_transaction(session, debit_tx.id, test_workspace.id, test_user.id, data)
    assert updated.description == "Updated Xfer"

    paired = await get_transaction(session, credit_tx.id, test_workspace.id)
    assert paired.description == "Updated Xfer"


# ---------------------------------------------------------------------------
# delete_transaction — transfer cascade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_transfer_cascades(session: AsyncSession, test_user, test_workspace, txn_account):
    acct2 = Account(
        id=uuid.uuid4(), user_id=test_user.id, name="DelSav",
        type="savings", balance=Decimal("0"), currency="BRL",
    )
    session.add(acct2)
    await session.commit()

    debit_tx, credit_tx = await create_transfer(session, test_workspace.id, test_user.id, TransferCreate(
        from_account_id=txn_account.id, to_account_id=acct2.id,
        description="Del Xfer", amount=Decimal("300"), date=date.today(),
    ))
    assert await delete_transaction(session, debit_tx.id, test_workspace.id) is True
    assert await get_transaction(session, credit_tx.id, test_workspace.id) is None


# ---------------------------------------------------------------------------
# update_transaction — changing account_id (regression for #63)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_transaction_changes_account(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    other_account = Account(
        id=uuid.uuid4(), user_id=test_user.id, name="OtherAcc",
        type="checking", balance=Decimal("0"), currency="BRL",
    )
    session.add(other_account)
    await session.commit()

    txn = await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
        account_id=txn_account.id, description="Move me",
        amount=Decimal("42"), date=date.today(), type="debit",
    ))

    updated = await update_transaction(
        session, txn.id, test_workspace.id, test_user.id,
        TransactionUpdate(account_id=other_account.id),
    )
    assert updated is not None
    assert updated.account_id == other_account.id

    # Re-fetch to make sure the change was committed, not just set in memory.
    reloaded = await get_transaction(session, txn.id, test_workspace.id)
    assert reloaded.account_id == other_account.id


@pytest.mark.asyncio
async def test_update_transaction_rejects_foreign_account(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    from app.models.user import User
    import bcrypt as _bcrypt

    from app.services.workspace_service import create_personal_workspace_for_user

    other_user = User(
        id=uuid.uuid4(),
        email="other@example.com",
        hashed_password=_bcrypt.hashpw(b"x", _bcrypt.gensalt()).decode(),
        is_active=True, is_superuser=False, is_verified=True,
    )
    session.add(other_user)
    await session.commit()
    other_ws = await create_personal_workspace_for_user(session, other_user)
    await session.commit()
    foreign_account = Account(
        id=uuid.uuid4(), user_id=other_user.id, workspace_id=other_ws.id, name="ForeignAcc",
        type="checking", balance=Decimal("0"), currency="BRL",
    )
    session.add(foreign_account)
    await session.commit()

    txn = await create_transaction(session, test_workspace.id, test_user.id, TransactionCreate(
        account_id=txn_account.id, description="Stay put",
        amount=Decimal("10"), date=date.today(), type="debit",
    ))

    with pytest.raises(ValueError, match="Account not found"):
        await update_transaction(
            session, txn.id, test_workspace.id, test_user.id,
            TransactionUpdate(account_id=foreign_account.id),
        )


@pytest.mark.asyncio
async def test_update_transfer_rejects_collapsing_accounts(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    acct2 = Account(
        id=uuid.uuid4(), user_id=test_user.id, name="XferSav",
        type="savings", balance=Decimal("0"), currency="BRL",
    )
    session.add(acct2)
    await session.commit()

    debit_tx, credit_tx = await create_transfer(session, test_workspace.id, test_user.id, TransferCreate(
        from_account_id=txn_account.id, to_account_id=acct2.id,
        description="Xfer", amount=Decimal("100"), date=date.today(),
    ))

    # Moving the debit side to acct2 would put both legs of the transfer
    # in the same account, which is invalid.
    with pytest.raises(ValueError, match="same account"):
        await update_transaction(
            session, debit_tx.id, test_workspace.id, test_user.id,
            TransactionUpdate(account_id=acct2.id),
        )


# ---------------------------------------------------------------------------
# Tag filtering and bulk tag operations (issue #88)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_transactions_filters_tags_with_exact_match(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    """Filtering by `#test` must NOT match `#test2` — exact tag boundaries."""
    txns = [
        Transaction(
            id=uuid.uuid4(), user_id=test_user.id, account_id=txn_account.id,
            description=desc, amount=Decimal("10"), date=date(2026, 1, day),
            type="debit", source="manual", notes=notes,
            created_at=datetime.now(timezone.utc),
        )
        for day, desc, notes in [
            (1, "A", "spent on lunch #test"),
            (2, "B", "#test2 work meal"),
            (3, "C", "#TEST uppercase variant"),
            (4, "D", "no tag here"),
            (5, "E", "#test #hey both"),
        ]
    ]
    session.add_all(txns)
    await session.commit()

    matches, total, _ = await get_transactions(
        session, test_workspace.id, test_user.id, tags=["#test"]
    )
    descriptions = {tx.description for tx in matches}
    assert descriptions == {"A", "C", "E"}
    assert total == 3


@pytest.mark.asyncio
async def test_get_transactions_filters_multiple_tags_with_or(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    """Multiple `tags` are OR-combined: a row matches if it carries ANY
    of the requested tags — issue #88."""
    txns = [
        Transaction(
            id=uuid.uuid4(), user_id=test_user.id, account_id=txn_account.id,
            description=desc, amount=Decimal("10"), date=date(2026, 1, day),
            type="debit", source="manual", notes=notes,
            created_at=datetime.now(timezone.utc),
        )
        for day, desc, notes in [
            (1, "A", "#test only"),
            (2, "B", "#hey only"),
            (3, "C", "#test and #hey together"),
            (4, "D", "#unrelated"),
        ]
    ]
    session.add_all(txns)
    await session.commit()

    matches, _, _ = await get_transactions(
        session, test_workspace.id, test_user.id, tags=["#test", "#hey"]
    )
    assert {tx.description for tx in matches} == {"A", "B", "C"}


@pytest.mark.asyncio
async def test_bulk_add_tags_appends_to_each_transaction(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    """bulk_add_tags must add the given tags to each tx, skipping duplicates."""
    t1 = Transaction(
        id=uuid.uuid4(), user_id=test_user.id, account_id=txn_account.id,
        description="A", amount=Decimal("10"), date=date.today(),
        type="debit", source="manual", notes=None,
        created_at=datetime.now(timezone.utc),
    )
    t2 = Transaction(
        id=uuid.uuid4(), user_id=test_user.id, account_id=txn_account.id,
        description="B", amount=Decimal("10"), date=date.today(),
        type="debit", source="manual", notes="existing #work note",
        created_at=datetime.now(timezone.utc),
    )
    t3 = Transaction(
        id=uuid.uuid4(), user_id=test_user.id, account_id=txn_account.id,
        description="C", amount=Decimal("10"), date=date.today(),
        type="debit", source="manual", notes="#groceries already here",
        created_at=datetime.now(timezone.utc),
    )
    session.add_all([t1, t2, t3])
    await session.commit()

    touched = await bulk_add_tags(
        session, test_workspace.id, [t1.id, t2.id, t3.id], ["#groceries"]
    )
    assert touched == 2  # t3 already has it

    await session.refresh(t1)
    await session.refresh(t2)
    await session.refresh(t3)
    assert t1.notes == "#groceries"
    assert t2.notes == "existing #work note #groceries"
    assert t3.notes == "#groceries already here"


@pytest.mark.asyncio
async def test_bulk_remove_tags_clears_only_exact_matches(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    """Removing `#test` must NOT touch `#test2` — exact match boundary."""
    t1 = Transaction(
        id=uuid.uuid4(), user_id=test_user.id, account_id=txn_account.id,
        description="A", amount=Decimal("10"), date=date.today(),
        type="debit", source="manual", notes="#test #keep",
        created_at=datetime.now(timezone.utc),
    )
    t2 = Transaction(
        id=uuid.uuid4(), user_id=test_user.id, account_id=txn_account.id,
        description="B", amount=Decimal("10"), date=date.today(),
        type="debit", source="manual", notes="#test2 untouched",
        created_at=datetime.now(timezone.utc),
    )
    session.add_all([t1, t2])
    await session.commit()

    touched = await bulk_remove_tags(
        session, test_workspace.id, [t1.id, t2.id], ["#test"]
    )
    assert touched == 1

    await session.refresh(t1)
    await session.refresh(t2)
    assert t1.notes == "#keep"
    assert t2.notes == "#test2 untouched"


# ---------------------------------------------------------------------------
# split_transaction_into_parts / revert_split_transaction
# ---------------------------------------------------------------------------


async def _mk_splittable_tx(
    session: AsyncSession,
    test_user,
    account,
    *,
    amount: str = "60.00",
    type_: str = "credit",
    amount_primary: Decimal | None = None,
    fx_rate_used: Decimal | None = None,
    date: date = date(2026, 5, 1),
    source: str = "manual",
    **kw,
) -> Transaction:
    tx = Transaction(
        id=uuid.uuid4(),
        user_id=test_user.id,
        account_id=account.id,
        description="Loan repayment + interest",
        amount=Decimal(amount),
        currency=account.currency,
        date=date,
        type=type_,
        source=source,
        amount_primary=amount_primary,
        fx_rate_used=fx_rate_used,
        created_at=datetime.now(timezone.utc),
        **kw,
    )
    session.add(tx)
    await session.commit()
    await session.refresh(tx)
    return tx


async def _setup_group_with_settlement_target(
    session: AsyncSession, test_user, test_workspace
):
    """A group with 2 members, for constructing a GroupSettlement/TransactionSplit
    row directly (bypassing settlement_service, which needs a full linked-member
    setup we don't need here — we only need the FK targets to be real rows)."""
    group = await group_service.create_group(
        session, test_workspace.id, test_user.id,
        GroupCreate(name=f"G-{uuid.uuid4().hex[:6]}", default_currency="BRL"),
    )
    me = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Me", is_self=True),
    )
    other = await group_service.create_member(
        session, group.id, test_workspace.id, GroupMemberCreate(name="Other", is_self=False),
    )
    return group, me, other


@pytest.mark.asyncio
async def test_split_transaction_into_parts_happy(
    session: AsyncSession, test_user, test_workspace, test_categories, txn_account
):
    original = await _mk_splittable_tx(
        session, test_user, txn_account,
        amount="60.00", amount_primary=Decimal("300.00"), fx_rate_used=Decimal("5.0"),
    )

    parent, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [
            TransactionSplitPartInput(amount=Decimal("40.00"), category_id=test_categories[0].id, description="Principal"),
            TransactionSplitPartInput(amount=Decimal("20.00"), category_id=test_categories[2].id, description="Interest"),
        ],
    )

    assert parent.is_ignored is True
    assert len(children) == 2
    assert sum(c.amount for c in children) == Decimal("60.00")
    for child in children:
        assert child.parent_transaction_id == original.id
        assert child.source == "manual"
        assert child.account_id == original.account_id
        assert child.date == original.date
        assert child.effective_date == original.effective_date
        assert child.type == original.type
        assert child.currency == original.currency
    assert children[0].description == "Principal"
    assert children[1].description == "Interest"
    # amount_primary proportional to each part, residual on the last child
    assert children[0].amount_primary == Decimal("200.00")
    assert children[1].amount_primary == Decimal("100.00")
    assert children[0].fx_rate_used == Decimal("5.0")
    assert sum(c.amount_primary for c in children) == parent.amount_primary


@pytest.mark.asyncio
async def test_split_transaction_into_parts_no_amount_primary(
    session: AsyncSession, test_user, test_workspace, test_categories, txn_account
):
    """When the parent has no amount_primary, children get None too."""
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")

    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [
            TransactionSplitPartInput(amount=Decimal("40.00")),
            TransactionSplitPartInput(amount=Decimal("20.00")),
        ],
    )
    for child in children:
        assert child.amount_primary is None


@pytest.mark.asyncio
async def test_split_transaction_into_parts_default_description(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    """A part without its own description falls back to the parent's."""
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")

    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [
            TransactionSplitPartInput(amount=Decimal("40.00")),
            TransactionSplitPartInput(amount=Decimal("20.00"), description="Interest"),
        ],
    )
    assert children[0].description == original.description
    assert children[1].description == "Interest"


@pytest.mark.asyncio
async def test_split_transaction_into_parts_too_few_parts(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    with pytest.raises(ValueError, match="between 2 and 20"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id,
            [TransactionSplitPartInput(amount=Decimal("60.00"))],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_too_many_parts(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="21.00")
    parts = [TransactionSplitPartInput(amount=Decimal("1.00")) for _ in range(21)]
    with pytest.raises(ValueError, match="between 2 and 20"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id, parts,
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_wrong_sum(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    with pytest.raises(ValueError, match="add up"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id,
            [
                TransactionSplitPartInput(amount=Decimal("40.00")),
                TransactionSplitPartInput(amount=Decimal("15.00")),
            ],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_nonpositive_amount(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    with pytest.raises(ValueError, match="positive"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id,
            [
                TransactionSplitPartInput(amount=Decimal("60.00")),
                TransactionSplitPartInput(amount=Decimal("0")),
            ],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_already_split(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    with pytest.raises(ValueError, match="already split"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id,
            [TransactionSplitPartInput(amount=Decimal("30.00")), TransactionSplitPartInput(amount=Decimal("30.00"))],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_rejects_a_child(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    with pytest.raises(ValueError, match="already part of a split"):
        await split_transaction_into_parts(
            session, children[0].id, test_workspace.id, test_user.id,
            [TransactionSplitPartInput(amount=Decimal("20.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_rejects_transfer_leg(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(
        session, test_user, txn_account, amount="60.00", transfer_pair_id=uuid.uuid4(),
    )
    with pytest.raises(ValueError, match="transfer leg"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id,
            [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_rejects_ignored(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00", is_ignored=True)
    with pytest.raises(ValueError, match="ignored"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id,
            [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_rejects_settlement_source(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00", source="settlement")
    with pytest.raises(ValueError, match="settlement transaction"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id,
            [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_rejects_settlement_link(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    group, me, other = await _setup_group_with_settlement_target(session, test_user, test_workspace)
    settlement = GroupSettlement(
        id=uuid.uuid4(), group_id=group.id, workspace_id=test_workspace.id,
        from_member_id=me.id, to_member_id=other.id, amount=Decimal("60.00"),
        currency="BRL", date=date(2026, 5, 1), transaction_id=original.id,
    )
    session.add(settlement)
    await session.commit()

    with pytest.raises(ValueError, match="settlement"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id,
            [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_rejects_group_splits(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    group, me, other = await _setup_group_with_settlement_target(session, test_user, test_workspace)
    split = TransactionSplit(
        id=uuid.uuid4(), transaction_id=original.id, workspace_id=test_workspace.id,
        group_member_id=other.id, share_amount=Decimal("30.00"), share_type="exact",
    )
    session.add(split)
    await session.commit()

    with pytest.raises(ValueError, match="group splits"):
        await split_transaction_into_parts(
            session, original.id, test_workspace.id, test_user.id,
            [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
        )


@pytest.mark.asyncio
async def test_split_transaction_into_parts_not_found(
    session: AsyncSession, test_user, test_workspace,
):
    with pytest.raises(ValueError, match="not found"):
        await split_transaction_into_parts(
            session, uuid.uuid4(), test_workspace.id, test_user.id,
            [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
        )


@pytest.mark.asyncio
async def test_revert_split_transaction_happy(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    parent, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    assert parent.is_ignored is True

    reverted = await revert_split_transaction(session, original.id, test_workspace.id)
    assert reverted.is_ignored is False

    for child in children:
        assert await get_transaction(session, child.id, test_workspace.id) is None
    assert await _has_split_children_helper(session, original.id) is False


async def _has_split_children_helper(session: AsyncSession, transaction_id) -> bool:
    from app.services.transaction_service import _has_split_children
    return await _has_split_children(session, transaction_id)


@pytest.mark.asyncio
async def test_revert_split_transaction_not_split(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    with pytest.raises(ValueError, match="not split"):
        await revert_split_transaction(session, original.id, test_workspace.id)


@pytest.mark.asyncio
async def test_revert_split_transaction_refuses_settlement_on_child(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    group, me, other = await _setup_group_with_settlement_target(session, test_user, test_workspace)
    settlement = GroupSettlement(
        id=uuid.uuid4(), group_id=group.id, workspace_id=test_workspace.id,
        from_member_id=me.id, to_member_id=other.id, amount=Decimal("20.00"),
        currency="BRL", date=date(2026, 5, 1), transaction_id=children[1].id,
    )
    session.add(settlement)
    await session.commit()

    with pytest.raises(ValueError, match="settlement"):
        await revert_split_transaction(session, original.id, test_workspace.id)


@pytest.mark.asyncio
async def test_revert_split_transaction_refuses_group_split_on_child(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    group, me, other = await _setup_group_with_settlement_target(session, test_user, test_workspace)
    split = TransactionSplit(
        id=uuid.uuid4(), transaction_id=children[0].id, workspace_id=test_workspace.id,
        group_member_id=other.id, share_amount=Decimal("20.00"), share_type="exact",
    )
    session.add(split)
    await session.commit()

    with pytest.raises(ValueError, match="group split"):
        await revert_split_transaction(session, original.id, test_workspace.id)


@pytest.mark.asyncio
async def test_revert_split_transaction_refuses_transfer_on_child(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    children[0].transfer_pair_id = uuid.uuid4()
    await session.commit()

    with pytest.raises(ValueError, match="transfer"):
        await revert_split_transaction(session, original.id, test_workspace.id)


# ---------------------------------------------------------------------------
# split-into-parts guards on delete/toggle_ignore/update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_transaction_blocks_split_parent(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    with pytest.raises(ValueError, match="split"):
        await delete_transaction(session, original.id, test_workspace.id)


@pytest.mark.asyncio
async def test_delete_transaction_blocks_split_child(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    with pytest.raises(ValueError, match="split"):
        await delete_transaction(session, children[0].id, test_workspace.id)


@pytest.mark.asyncio
async def test_toggle_ignore_blocks_split_parent(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    with pytest.raises(ValueError, match="split"):
        await toggle_ignore_transaction(session, original.id, test_workspace.id)


@pytest.mark.asyncio
async def test_toggle_ignore_blocks_split_child(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    with pytest.raises(ValueError, match="split"):
        await toggle_ignore_transaction(session, children[0].id, test_workspace.id)


@pytest.mark.asyncio
async def test_update_transaction_blocks_locked_fields_on_split_child(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    with pytest.raises(ValueError, match="split"):
        await update_transaction(
            session, children[0].id, test_workspace.id, test_user.id,
            TransactionUpdate(amount=Decimal("41.00")),
        )


@pytest.mark.asyncio
async def test_update_transaction_blocks_locked_fields_on_split_parent(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    with pytest.raises(ValueError, match="split"):
        await update_transaction(
            session, original.id, test_workspace.id, test_user.id,
            TransactionUpdate(date=date(2026, 6, 1)),
        )


@pytest.mark.asyncio
async def test_update_transaction_allows_free_fields_on_split_child(
    session: AsyncSession, test_user, test_workspace, test_categories, txn_account
):
    """category_id/description/notes/payee stay editable on a split part."""
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    updated = await update_transaction(
        session, children[0].id, test_workspace.id, test_user.id,
        TransactionUpdate(description="Renamed part", category_id=test_categories[1].id, notes="a note"),
    )
    assert updated is not None
    assert updated.description == "Renamed part"
    assert updated.category_id == test_categories[1].id
    assert updated.notes == "a note"


@pytest.mark.asyncio
async def test_update_transaction_resubmitting_unchanged_is_ignored_does_not_block(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    """The edit form always echoes the current is_ignored value in the save
    payload. Resubmitting the SAME value must not trip the split guard."""
    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00")
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )
    child = children[0]
    updated = await update_transaction(
        session, child.id, test_workspace.id, test_user.id,
        TransactionUpdate(description="Still fine", is_ignored=child.is_ignored),
    )
    assert updated is not None
    assert updated.description == "Still fine"


# ---------------------------------------------------------------------------
# split-into-parts shielding: fuzzy match / transfer detection / P&L
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fuzzy_match_manual_excludes_split_child(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    original = await _mk_splittable_tx(
        session, test_user, txn_account, amount="40.00", type_="debit",
        date=date(2026, 5, 10),
    )
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("25.00")), TransactionSplitPartInput(amount=Decimal("15.00"))],
    )
    child = children[0]
    child.description = "STARBUCKS COFFEE"
    await session.commit()

    txn_data = TransactionData(
        external_id="ext-1", description="STARBUCKS COFFEE",
        amount=Decimal("25.00"), date=date(2026, 5, 10), type="debit",
    )
    match = await _fuzzy_match_manual(session, txn_account.id, txn_data)
    assert match is None


@pytest.mark.asyncio
async def test_detect_transfer_pairs_excludes_ignored_parent_and_child(
    session: AsyncSession, test_user, test_workspace, txn_account
):
    other_account = Account(
        id=uuid.uuid4(), user_id=test_user.id, name="Other",
        type="checking", balance=Decimal("0"), currency="BRL",
    )
    session.add(other_account)
    await session.commit()

    original = await _mk_splittable_tx(
        session, test_user, txn_account, amount="60.00", type_="debit",
        date=date(2026, 5, 10),
    )
    _, children = await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [TransactionSplitPartInput(amount=Decimal("40.00")), TransactionSplitPartInput(amount=Decimal("20.00"))],
    )

    # A same-amount, same-date credit in another account would normally
    # pair as a transfer with either the (now-ignored) parent or a child.
    credit_for_parent = Transaction(
        id=uuid.uuid4(), user_id=test_user.id, account_id=other_account.id,
        description="Would-be pair for parent", amount=Decimal("60.00"),
        date=date(2026, 5, 10), type="credit", source="manual",
        created_at=datetime.now(timezone.utc),
    )
    credit_for_child = Transaction(
        id=uuid.uuid4(), user_id=test_user.id, account_id=other_account.id,
        description="Would-be pair for child", amount=Decimal("40.00"),
        date=date(2026, 5, 10), type="credit", source="manual",
        created_at=datetime.now(timezone.utc),
    )
    session.add_all([credit_for_parent, credit_for_child])
    await session.commit()

    pairs_created = await detect_transfer_pairs(session, test_workspace.id)
    assert pairs_created == 0

    await session.refresh(original)
    for child in children:
        await session.refresh(child)
    assert original.transfer_pair_id is None
    assert all(c.transfer_pair_id is None for c in children)
    assert credit_for_parent.transfer_pair_id is None
    assert credit_for_child.transfer_pair_id is None


@pytest.mark.asyncio
async def test_split_parts_pnl_counts_only_non_transfer_child(
    session: AsyncSession, test_user, test_workspace, test_categories, txn_account
):
    """A split with one part in a treat_as_transfer category and one in a
    normal category: only the normal one counts toward P&L, and the
    original (ignored) parent counts toward neither."""
    transfer_category = Category(
        id=uuid.uuid4(), user_id=test_user.id, workspace_id=test_workspace.id,
        name="Loan principal", treat_as_transfer=True,
    )
    session.add(transfer_category)
    await session.commit()

    original = await _mk_splittable_tx(session, test_user, txn_account, amount="60.00", type_="credit")
    await split_transaction_into_parts(
        session, original.id, test_workspace.id, test_user.id,
        [
            TransactionSplitPartInput(amount=Decimal("40.00"), category_id=transfer_category.id),
            TransactionSplitPartInput(amount=Decimal("20.00"), category_id=test_categories[2].id),
        ],
    )

    _, _, summary = await get_transactions(
        session, test_workspace.id, test_user.id, include_summary=True
    )
    assert summary["income"] == Decimal("20.00")
    assert summary["excluded"] == Decimal("100.00")  # 60 (ignored parent) + 40 (transfer child)
