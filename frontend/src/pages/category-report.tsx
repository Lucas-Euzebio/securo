import { Fragment, useState } from 'react'
import { CalendarIcon, ChevronDown, ChevronRight, ChevronsDownUp, ChevronsUpDown } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { currentMonth, shiftMonth, monthRange, monthLabel } from '@/lib/month-utils'
import { dashboard, budgets, transactions as transactionsApi, accounts as accountsApi, categories as categoriesApi, categoryGroups as categoryGroupsApi } from '@/lib/api'
import { invalidateFinancialQueries } from '@/lib/invalidate-queries'
import { getAccountName } from '@/lib/account-utils'
import { Skeleton } from '@/components/ui/skeleton'
import { Popover, PopoverTrigger, PopoverContent } from '@/components/ui/popover'
import { MonthPicker } from '@/components/ui/monthpicker'
import { resolveDateFnsLocale } from '@/lib/date-fns-locale'
import { format } from 'date-fns'
import { PageHeader } from '@/components/page-header'
import { CategoryIcon } from '@/components/category-icon'
import { TransactionDrillDown, type DrillDownFilter } from '@/components/transaction-drill-down'
import { TransactionDialog, extractTxApiError } from '@/components/transaction-dialog'
import { usePrivacyMode } from '@/hooks/use-privacy-mode'
import { useAuth } from '@/contexts/auth-context'
import { useCollectionFilter } from '@/contexts/collection-filter-context'
import { useDisplayLocale } from '@/hooks/use-display-locale'
import type { SpendingByCategory, Transaction } from '@/types'

const RANGE_PRESETS = [3, 6, 12, 24] as const
// Each month in view costs two requests (spending + budgets) — keep a sane cap.
const MAX_MONTHS = 36

function monthDiff(a: string, b: string): number {
  const [ay, am] = a.split('-').map(Number)
  const [by, bm] = b.split('-').map(Number)
  return (by - ay) * 12 + (bm - am)
}

type CategoryRow = {
  id: string | null
  name: string
  icon: string
  color: string
  values: number[]
  total: number
  avg: number
  budget: number | null
}

type GroupBlock = {
  id: string
  name: string
  rows: CategoryRow[]
  subtotal: number[]
  total: number
  avg: number
  budget: number
  budgetByMonth: number[]
}

function formatCurrency(value: number, currency = 'USD', locale = 'en-US') {
  return new Intl.NumberFormat(locale, { style: 'currency', currency, maximumFractionDigits: 0 }).format(value)
}

/**
 * Diverging tint vs a reference (the budget when one is set, else the row's
 * own monthly average): teal below the reference, rose above, fading to
 * neutral at the reference itself. Teal instead of pure green keeps the two
 * poles apart under red-green color blindness (deutan ΔE 10.1 vs 5.6); the
 * number in the cell is the secondary encoding.
 */
function heatStyle(value: number, reference: number | null): React.CSSProperties | undefined {
  if (value <= 0 || !reference || reference <= 0) return undefined
  const deviation = (value - reference) / reference
  if (Math.abs(deviation) < 0.02) return undefined
  const intensity = Math.min(Math.abs(deviation), 1)
  const alpha = (0.06 + 0.3 * intensity).toFixed(3)
  return {
    backgroundColor: deviation > 0
      ? `rgba(244, 63, 94, ${alpha})`
      : `rgba(20, 184, 166, ${alpha})`,
  }
}

export default function CategoryReportPage() {
  const { t, i18n } = useTranslation()
  const { user } = useAuth()
  const { mask } = usePrivacyMode()
  const locale = useDisplayLocale()
  const uiLocale = i18n.resolvedLanguage ?? i18n.language
  const dateFnsLocale = resolveDateFnsLocale(uiLocale)
  const userCurrency = user?.preferences?.currency_display ?? 'USD'

  const thisMonthInit = currentMonth()
  const [fromMonth, setFromMonth] = useState<string>(shiftMonth(thisMonthInit, -11))
  const [toMonth, setToMonth] = useState<string>(thisMonthInit)
  const [fromOpen, setFromOpen] = useState(false)
  const [toOpen, setToOpen] = useState(false)
  const [includeCurrentMonth, setIncludeCurrentMonth] = useState(false)
  const [drillDown, setDrillDown] = useState<DrillDownFilter | null>(null)
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())
  const [editingTx, setEditingTx] = useState<Transaction | null>(null)
  const queryClient = useQueryClient()

  const onMutationSuccess = () => {
    invalidateFinancialQueries(queryClient)
    queryClient.invalidateQueries({ queryKey: ['category-report'] })
    setEditingTx(null)
  }
  const updateMutation = useMutation({
    mutationFn: ({ id, ...data }: Partial<Transaction> & { id: string }) => transactionsApi.update(id, data),
    onSuccess: onMutationSuccess,
  })
  const deleteMutation = useMutation({
    mutationFn: (id: string) => transactionsApi.delete(id),
    onSuccess: onMutationSuccess,
  })

  const toggleGroup = (groupId: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(groupId)) next.delete(groupId)
      else next.add(groupId)
      return next
    })
  }

  const { activeAccountIds } = useCollectionFilter()
  const acctIds = activeAccountIds ?? undefined
  const noAccounts = activeAccountIds !== null && activeAccountIds.length === 0

  const thisMonth = currentMonth()
  // Oldest → newest across the selected window (inclusive on both ends).
  const span = Math.min(Math.max(monthDiff(fromMonth, toMonth), 0), MAX_MONTHS - 1)
  const months = Array.from({ length: span + 1 }, (_, i) => shiftMonth(fromMonth, i))

  // Month indices the Média/mês columns average over: the current (partial)
  // month is left out unless the toggle includes it — the column itself stays
  // visible in the table either way. A window that is ONLY the current month
  // falls back to including it (an empty average makes no sense).
  let avgIndexes = months.map((_, i) => i).filter((i) => includeCurrentMonth || months[i] !== thisMonth)
  if (avgIndexes.length === 0) avgIndexes = months.map((_, i) => i)

  const setPreset = (n: number) => {
    setFromMonth(shiftMonth(thisMonth, -(n - 1)))
    setToMonth(thisMonth)
  }
  const activePreset = RANGE_PRESETS.find(
    (n) => toMonth === thisMonth && fromMonth === shiftMonth(thisMonth, -(n - 1)),
  )

  const spendingQueries = useQueries({
    queries: months.map((m) => ({
      queryKey: ['category-report', 'spending', m, activeAccountIds],
      queryFn: () => dashboard.spendingByCategory(monthRange(m).from, acctIds),
      enabled: !noAccounts,
      staleTime: 5 * 60 * 1000,
    })),
  })

  // One comparison call per month: budgets are month-scoped (recurring
  // defaults + monthly overrides), so each column gets its own effective
  // budget instead of stretching the current month's plan over the past.
  const budgetQueries = useQueries({
    queries: months.map((m) => ({
      queryKey: ['category-report', 'budget-comparison', m],
      queryFn: () => budgets.comparison(monthRange(m).from),
      staleTime: 5 * 60 * 1000,
    })),
  })
  const { data: categoriesList } = useQuery({ queryKey: ['categories'], queryFn: categoriesApi.list })
  const { data: groupsList } = useQuery({ queryKey: ['category-groups'], queryFn: categoryGroupsApi.list })
  const { data: accountsList } = useQuery({ queryKey: ['accounts'], queryFn: () => accountsApi.list(), enabled: editingTx !== null })

  const isLoading = spendingQueries.some((q) => q.isLoading)

  // ---- Build the category × month matrix -------------------------------
  // Budgets may be stored with a debit sign (negative) — treat as magnitude.
  const budgetsPerMonth = budgetQueries.map((q) => {
    const map = new Map<string, number>()
    for (const row of q.data ?? []) {
      const amount = Math.abs(row.budget_amount ?? 0)
      if (amount > 0) map.set(row.category_id, amount)
    }
    return map
  })
  // The Budget column shows the average effective budget across the selected
  // window, counting only months that actually had a plan — anchored to the
  // period under analysis, never to today's date. Months without a budget are
  // "no plan", not "plan = 0", so they don't dilute the average.
  const avgBudgetByCategory = new Map<string, number>()
  {
    const sums = new Map<string, { sum: number; n: number }>()
    for (const monthMap of budgetsPerMonth) {
      for (const [id, amount] of monthMap) {
        const entry = sums.get(id) ?? { sum: 0, n: 0 }
        entry.sum += amount
        entry.n += 1
        sums.set(id, entry)
      }
    }
    for (const [id, entry] of sums) avgBudgetByCategory.set(id, entry.sum / entry.n)
  }
  const groupIdByCategory = new Map<string, string | null>()
  for (const cat of categoriesList ?? []) groupIdByCategory.set(cat.id, cat.group_id)
  const groupNameById = new Map<string, string>()
  for (const g of groupsList ?? []) groupNameById.set(g.id, g.name)

  const rowsByKey = new Map<string, CategoryRow>()
  spendingQueries.forEach((q, monthIdx) => {
    for (const item of (q.data ?? []) as SpendingByCategory[]) {
      const key = item.category_id ?? '__uncategorized__'
      let row = rowsByKey.get(key)
      if (!row) {
        row = {
          id: item.category_id,
          name: item.category_name,
          icon: item.category_icon,
          color: item.category_color,
          values: months.map(() => 0),
          total: 0,
          avg: 0,
          budget: item.category_id ? avgBudgetByCategory.get(item.category_id) ?? null : null,
        }
        rowsByKey.set(key, row)
      }
      row.values[monthIdx] += item.total
    }
  })

  const UNGROUPED = '__ungrouped__'
  const blocksById = new Map<string, GroupBlock>()
  for (const row of rowsByKey.values()) {
    row.total = row.values.reduce((s, v) => s + v, 0)
    row.avg = avgIndexes.reduce((s, i) => s + row.values[i], 0) / avgIndexes.length
    const groupId = (row.id ? groupIdByCategory.get(row.id) : null) ?? UNGROUPED
    let block = blocksById.get(groupId)
    if (!block) {
      block = {
        id: groupId,
        name: groupId === UNGROUPED ? t('categoryReport.ungrouped') : groupNameById.get(groupId) ?? t('categoryReport.ungrouped'),
        rows: [],
        subtotal: months.map(() => 0),
        total: 0,
        avg: 0,
        budget: 0,
        budgetByMonth: months.map(() => 0),
      }
      blocksById.set(groupId, block)
    }
    block.rows.push(row)
    block.budget += row.budget ?? 0
    row.values.forEach((v, i) => {
      block!.subtotal[i] += v
      if (row.id) block!.budgetByMonth[i] += budgetsPerMonth[i]?.get(row.id) ?? 0
    })
  }
  const blocks = [...blocksById.values()]
  for (const block of blocks) {
    block.total = block.subtotal.reduce((s, v) => s + v, 0)
    block.avg = avgIndexes.reduce((s, i) => s + block.subtotal[i], 0) / avgIndexes.length
    block.rows.sort((a, b) => b.total - a.total)
  }
  // Biggest spending groups first; the ungrouped bucket always last.
  blocks.sort((a, b) => (a.id === UNGROUPED ? 1 : b.id === UNGROUPED ? -1 : b.total - a.total))

  const allCollapsed = blocks.length > 0 && blocks.every((b) => collapsed.has(b.id))
  const toggleAll = () => setCollapsed(allCollapsed ? new Set() : new Set(blocks.map((b) => b.id)))

  const grandTotal = months.map((_, i) => blocks.reduce((s, b) => s + b.subtotal[i], 0))
  const grandSum = grandTotal.reduce((s, v) => s + v, 0)
  const totalBudget = blocks.reduce((s, b) => s + b.budget, 0)
  const totalBudgetByMonth = months.map((_, i) => blocks.reduce((s, b) => s + b.budgetByMonth[i], 0))

  const monthHeader = (m: string) => {
    const [y, mo] = m.split('-').map(Number)
    return new Date(y, mo - 1, 2).toLocaleDateString(uiLocale, { month: 'short', year: '2-digit' })
  }

  const openDrillDown = (row: CategoryRow, monthIdx: number) => {
    const m = months[monthIdx]
    const { from, to } = monthRange(m)
    setDrillDown({
      title: `${row.name} — ${monthLabel(m, uiLocale)}`,
      category_id: row.id ?? undefined,
      uncategorized: row.id === null || undefined,
      account_ids: acctIds,
      type: 'debit',
      from,
      to,
    })
  }

  const fmt = (v: number) => mask(formatCurrency(v, userCurrency, locale))
  const cellClass = 'px-2 py-1.5 text-right tabular-nums whitespace-nowrap'

  return (
    <div className="max-w-full">
      <PageHeader
        section={t('nav.groupAnalysis')}
        title={t('categoryReport.title')}
        action={
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={toggleAll}
              title={allCollapsed ? t('categoryReport.expandAll') : t('categoryReport.collapseAll')}
              className="flex items-center justify-center w-7 h-7 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
            >
              {allCollapsed ? <ChevronsUpDown size={15} /> : <ChevronsDownUp size={15} />}
            </button>
            <div className="flex items-center gap-1 bg-muted rounded-lg p-0.5">
              {RANGE_PRESETS.map((n) => (
                <button
                  key={n}
                  onClick={() => setPreset(n)}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer ${
                    activePreset === n ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {n}M
                </button>
              ))}
            </div>
            <label
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border border-border bg-card text-foreground cursor-pointer hover:bg-muted transition-colors select-none"
              title={t('categoryReport.includeCurrentMonthTooltip')}
            >
              <input
                type="checkbox"
                checked={includeCurrentMonth}
                onChange={(e) => setIncludeCurrentMonth(e.target.checked)}
                className="h-3.5 w-3.5 rounded border-border accent-primary"
              />
              {t('categoryReport.includeCurrentMonth')}
            </label>
            <div className="flex items-center gap-1">
              <Popover open={fromOpen} onOpenChange={setFromOpen}>
                <PopoverTrigger asChild>
                  <button className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border border-border bg-card text-foreground hover:bg-muted transition-colors cursor-pointer">
                    <CalendarIcon size={13} className="text-muted-foreground" />
                    {t('categoryReport.from')} {monthHeader(fromMonth)}
                  </button>
                </PopoverTrigger>
                <PopoverContent align="end" className="w-auto p-0">
                  <MonthPicker
                    locale={dateFnsLocale}
                    selectedMonth={new Date(`${fromMonth}-01T00:00:00`)}
                    maxDate={new Date(`${thisMonth}-01T00:00:00`)}
                    onMonthSelect={(date) => {
                      if (!date) return
                      const m = format(date, 'yyyy-MM')
                      setFromMonth(m)
                      if (m > toMonth) setToMonth(m)
                      setFromOpen(false)
                    }}
                  />
                </PopoverContent>
              </Popover>
              <Popover open={toOpen} onOpenChange={setToOpen}>
                <PopoverTrigger asChild>
                  <button className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border border-border bg-card text-foreground hover:bg-muted transition-colors cursor-pointer">
                    <CalendarIcon size={13} className="text-muted-foreground" />
                    {t('categoryReport.to')} {monthHeader(toMonth)}
                  </button>
                </PopoverTrigger>
                <PopoverContent align="end" className="w-auto p-0">
                  <MonthPicker
                    locale={dateFnsLocale}
                    selectedMonth={new Date(`${toMonth}-01T00:00:00`)}
                    maxDate={new Date(`${thisMonth}-01T00:00:00`)}
                    onMonthSelect={(date) => {
                      if (!date) return
                      const m = format(date, 'yyyy-MM')
                      setToMonth(m)
                      if (m < fromMonth) setFromMonth(m)
                      setToOpen(false)
                    }}
                  />
                </PopoverContent>
              </Popover>
            </div>
          </div>
        }
      />
      <p className="text-sm text-muted-foreground -mt-4 mb-5">{t('categoryReport.subtitle')}</p>

      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-9 w-full" />
          ))}
        </div>
      ) : rowsByKey.size === 0 ? (
        <div className="bg-card rounded-xl border border-border p-12 text-center text-sm text-muted-foreground">
          {t('categoryReport.empty')}
        </div>
      ) : (
        <div className="bg-card rounded-xl border border-border shadow-sm overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-border">
                <th className="sticky left-0 bg-card px-3 py-2 text-left font-semibold text-muted-foreground min-w-[180px]">
                  {t('transactions.category')}
                </th>
                {months.map((m) => (
                  <th key={m} className="px-2 py-2 text-right font-medium text-muted-foreground whitespace-nowrap">
                    {monthHeader(m)}
                  </th>
                ))}
                <th className="px-2 py-2 text-right font-semibold text-foreground whitespace-nowrap border-l border-border">
                  {t('categoryReport.avg')}
                </th>
                <th className="px-2 py-2 text-right font-semibold text-foreground whitespace-nowrap">
                  {t('categoryReport.total')}
                </th>
                <th
                  className="px-2 py-2 text-right font-semibold text-foreground whitespace-nowrap cursor-help"
                  title={t('categoryReport.budgetTooltip')}
                >
                  {t('categoryReport.budget')}
                </th>
              </tr>
            </thead>
            <tbody>
              {blocks.map((block) => (
                <Fragment key={block.id}>
                  <tr
                    className="bg-muted/50 border-b border-border cursor-pointer hover:bg-muted"
                    onClick={() => toggleGroup(block.id)}
                  >
                    <td className="sticky left-0 bg-muted px-3 py-1.5 font-semibold text-foreground">
                      <div className="flex items-center gap-1.5">
                        {collapsed.has(block.id)
                          ? <ChevronRight size={13} className="shrink-0 text-muted-foreground" />
                          : <ChevronDown size={13} className="shrink-0 text-muted-foreground" />}
                        <span className="truncate">{block.name}</span>
                        {collapsed.has(block.id) && (
                          <span className="text-[10px] font-normal text-muted-foreground">({block.rows.length})</span>
                        )}
                      </div>
                    </td>
                    {block.subtotal.map((v, i) => (
                      <td key={i} className={`${cellClass} font-semibold text-foreground`} style={heatStyle(v, block.budgetByMonth[i] > 0 ? block.budgetByMonth[i] : null)}>
                        {v > 0 ? fmt(v) : ''}
                      </td>
                    ))}
                    <td className={`${cellClass} font-semibold border-l border-border`}>{fmt(block.avg)}</td>
                    <td className={`${cellClass} font-semibold`}>{fmt(block.total)}</td>
                    <td className={`${cellClass} font-semibold`}>{block.budget > 0 ? fmt(block.budget) : ''}</td>
                  </tr>
                  {!collapsed.has(block.id) && block.rows.map((row) => {
                    return (
                      <tr key={`${block.id}:${row.id ?? 'uncategorized'}`} className="border-b border-border/50 hover:bg-muted/30">
                        <td className="sticky left-0 bg-card px-3 py-1.5">
                          <div className="flex items-center gap-2 min-w-0">
                            <CategoryIcon icon={row.icon} color={row.color} size="sm" />
                            <span className="truncate text-foreground">{row.name}</span>
                          </div>
                        </td>
                        {row.values.map((v, i) => (
                          <td
                            key={i}
                            className={`${cellClass} ${v > 0 ? 'cursor-pointer text-foreground' : 'text-muted-foreground/40'}`}
                            style={heatStyle(v, (row.id ? budgetsPerMonth[i]?.get(row.id) : undefined) ?? row.avg)}
                            onClick={v > 0 ? () => openDrillDown(row, i) : undefined}
                          >
                            {v > 0 ? fmt(v) : '·'}
                          </td>
                        ))}
                        <td className={`${cellClass} font-medium border-l border-border`}>{fmt(row.avg)}</td>
                        <td className={`${cellClass} font-medium`}>{fmt(row.total)}</td>
                        <td className={`${cellClass} text-muted-foreground`}>{row.budget != null ? fmt(row.budget) : '—'}</td>
                      </tr>
                    )
                  })}
                </Fragment>
              ))}
              <tr className="border-t-2 border-border bg-muted/70">
                <td className="sticky left-0 bg-muted px-3 py-2 font-bold text-foreground">{t('categoryReport.total')}</td>
                {grandTotal.map((v, i) => (
                  <td key={i} className={`${cellClass} py-2 font-bold text-foreground`} style={heatStyle(v, totalBudgetByMonth[i] > 0 ? totalBudgetByMonth[i] : null)}>{fmt(v)}</td>
                ))}
                <td className={`${cellClass} py-2 font-bold border-l border-border`}>{fmt(avgIndexes.reduce((s, i) => s + grandTotal[i], 0) / avgIndexes.length)}</td>
                <td className={`${cellClass} py-2 font-bold`}>{fmt(grandSum)}</td>
                <td className={`${cellClass} py-2 font-bold`}>{fmt(totalBudget)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <TransactionDrillDown
        filter={drillDown}
        onClose={() => setDrillDown(null)}
        onTransactionClick={(tx) => setEditingTx(tx)}
      />

      <TransactionDialog
        open={editingTx !== null}
        onClose={() => setEditingTx(null)}
        transaction={editingTx}
        categories={categoriesList ?? []}
        categoryGroups={groupsList ?? []}
        accounts={(accountsList ?? []).map((a: { id: string; name: string; display_name?: string | null }) => ({ id: a.id, name: getAccountName(a) }))}
        onSave={(data) => {
          if (editingTx) updateMutation.mutate({ id: editingTx.id, ...data })
        }}
        onDelete={() => {
          if (editingTx) deleteMutation.mutate(editingTx.id)
        }}
        loading={updateMutation.isPending || deleteMutation.isPending}
        error={updateMutation.error ? extractTxApiError(updateMutation.error, t) : deleteMutation.error ? extractTxApiError(deleteMutation.error, t) : null}
        isSynced={editingTx?.source === 'sync'}
      />
    </div>
  )
}
