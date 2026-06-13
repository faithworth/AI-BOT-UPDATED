'use client';
// ============================================================
// VUKA — Admin Finance (full rebuild)
// Tabs: Overview · All Sales · Tips · Per-Artist · Payouts
// ============================================================

import { useEffect, useState, useCallback, useRef } from 'react';
import {
  Loader2, RefreshCw, CheckCircle, XCircle, DollarSign,
  TrendingUp, Download, Clock, ChevronRight, Search,
  ArrowLeft, Music, Film, Mic2, Package, Tag, Heart,
  User, AlertTriangle,
} from 'lucide-react';
import { formatCurrency } from '@/lib/utils';

type MainTab = 'overview' | 'sales' | 'tips' | 'subscriptions' | 'artists' | 'payouts';

const STATUS_COLORS: Record<string, string> = {
  confirmed: '#a0e87c', pending: '#e8c87c', refunded: '#e8a87c',
  PENDING: '#e8c87c', APPROVED: '#38b6e8', PROCESSING: '#38b6e8',
  COMPLETED: '#a0e87c', PAID: '#a0e87c', paid: '#a0e87c',
  FAILED: '#ff4d4d', CANCELLED: '#a0a0a0', rejected: '#ff4d4d',
};

const TYPE_ICONS: Record<string, any> = {
  beat: Music, release: Tag, video: Film, sample: Package,
  subscription: Mic2, merch: Package,
};

function Pill({ label, color }: { label: string; color?: string }) {
  const c = color || STATUS_COLORS[label] || '#a0a0a0';
  return (
    <span className="px-2 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wide"
      style={{ background: `${c}22`, color: c }}>
      {label}
    </span>
  );
}

function StatCard({ label, value, sub, color, onClick }: {
  label: string; value: string; sub?: string; color?: string; onClick?: () => void;
}) {
  return (
    <div onClick={onClick}
      className="p-4 rounded-2xl flex flex-col gap-1"
      style={{
        background: 'var(--surface)', border: '1px solid var(--border)',
        cursor: onClick ? 'pointer' : undefined,
      }}>
      <div className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>{label}</div>
      <div className="text-xl font-black font-mono" style={{ color: color || 'var(--text)' }}>{value}</div>
      {sub && <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{sub}</div>}
    </div>
  );
}

// ─── Overview tab ──────────────────────────────────────────────────────────────
function OverviewTab({ onArtistClick }: { onArtistClick: (id: string) => void }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch('/api/admin/finance?view=overview')
      .then(r => r.json()).then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex items-center justify-center py-20"><Loader2 className="animate-spin" size={24} style={{ color: 'var(--green)' }} /></div>;
  if (!data) return null;

  const { revenue, payouts, topArtists, salesByType } = data;

  return (
    <div className="space-y-8">
      {/* Revenue summary */}
      <div>
        <p className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>All-time Revenue</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Total Gross" value={formatCurrency(revenue.gross)} color="var(--green)"
            sub={`${revenue.salesCount} sales · ${revenue.tipsCount} tips · ${revenue.subsCount ?? 0} subs`} />
          <StatCard label="Vuka Keeps" value={formatCurrency(revenue.platformCut)} color="var(--gold)" />
          <StatCard label="Artists Get" value={formatCurrency(revenue.artistTotal)} color="#38b6e8" />
          <StatCard label="Paid Out" value={formatCurrency(payouts.paidAmount)} color="#a0a0a0" sub={`${payouts.paidCount} payouts`} />
        </div>
      </div>

      <div>
        <p className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>Last 30 Days</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Gross (30d)" value={formatCurrency(revenue.monthGross)} color="var(--green)"
            sub={`${revenue.monthSalesCount} sales · ${revenue.monthTipsCount} tips · ${revenue.monthSubsCount ?? 0} subs`} />
          <StatCard label="Vuka Keeps (30d)" value={formatCurrency(revenue.monthPlatform)} color="var(--gold)" />
          <StatCard label="Plan Subscriptions" value={formatCurrency(revenue.subsTotal ?? 0)} color="#c084fc"
            sub={`${revenue.subsCount ?? 0} total`} />
          <StatCard label="Pending Payouts" value={formatCurrency(payouts.pendingAmount)} color="#e8a87c" sub={`${payouts.pendingCount} requests`} />
        </div>
      </div>

      {/* Sales by type */}
      {salesByType?.length > 0 && (
        <div>
          <p className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>Sales by Type</p>
          <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            <div className="overflow-x-auto">
              <table className="w-full text-sm" style={{ minWidth: 500 }}>
                <thead>
                  <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                    {['Type', 'Sales', 'Gross', 'Platform Fee', 'Artist Net'].map(h => (
                      <th key={h} className="px-4 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {salesByType.map((row: any) => {
                    const gross    = row._sum?.amount || 0;
                    const platform = row._sum?.platformFee || 0;
                    return (
                      <tr key={row.itemType} className="border-t" style={{ borderColor: 'var(--border)' }}>
                        <td className="px-4 py-3 font-semibold capitalize whitespace-nowrap">{row.itemType}</td>
                        <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{row._count}</td>
                        <td className="px-4 py-3 font-mono font-bold whitespace-nowrap" style={{ color: 'var(--green)' }}>{formatCurrency(gross)}</td>
                        <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--gold)' }}>{formatCurrency(platform)}</td>
                        <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: '#38b6e8' }}>{formatCurrency(gross - platform)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Top artists */}
      {topArtists?.length > 0 && (
        <div>
          <p className="text-xs font-bold uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>Top Artists by Revenue</p>
          <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            <div className="overflow-x-auto">
              <table className="w-full text-sm" style={{ minWidth: 700 }}>
                <thead>
                  <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                    {['Artist', 'Sales', 'Tips', 'Gross', 'Vuka Keeps', 'Artist Gets', 'Paid Out', 'Balance'].map(h => (
                      <th key={h} className="px-4 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {topArtists.map((row: any) => {
                    const balance = row.artistOwes - row.payoutsTotal;
                    return (
                      <tr key={row.artistId}
                        className="border-t hover:bg-white/[0.02] cursor-pointer"
                        style={{ borderColor: 'var(--border)' }}
                        onClick={() => onArtistClick(row.artistId)}>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <div className="flex items-center gap-2">
                            {row.artist?.photoUrl
                              ? <img src={row.artist.photoUrl} className="w-7 h-7 rounded-full object-cover flex-shrink-0" />
                              : <div className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center" style={{ background: 'var(--surface2)' }}><User size={12} style={{ color: 'var(--text-muted)' }} /></div>}
                            <span className="font-semibold">{row.artist?.name || row.artistId.slice(0, 8)}</span>
                            <ChevronRight size={12} style={{ color: 'var(--text-muted)' }} />
                          </div>
                        </td>
                        <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{row.salesCount}</td>
                        <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{row.tipsCount}</td>
                        <td className="px-4 py-3 font-mono font-bold whitespace-nowrap" style={{ color: 'var(--green)' }}>{formatCurrency(row.grossSales)}</td>
                        <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--gold)' }}>{formatCurrency(row.platformCut)}</td>
                        <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: '#38b6e8' }}>{formatCurrency(row.artistOwes)}</td>
                        <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{formatCurrency(row.payoutsTotal)}</td>
                        <td className="px-4 py-3 font-mono font-bold whitespace-nowrap"
                          style={{ color: balance > 0 ? '#e8c87c' : '#a0a0a0' }}>{formatCurrency(balance)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Sales tab ─────────────────────────────────────────────────────────────────
function SalesTab() {
  const [data, setData]       = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage]       = useState(1);
  const [search, setSearch]   = useState('');
  const [q, setQ]             = useState('');

  const load = useCallback(() => {
    setLoading(true);
    fetch(`/api/admin/finance?view=purchases&page=${page}&q=${encodeURIComponent(q)}`)
      .then(r => r.json()).then(setData).finally(() => setLoading(false));
  }, [page, q]);

  useEffect(() => { load(); }, [load]);

  function getItemTitle(p: any) {
    return p.beat?.title || p.release?.title || p.video?.title || p.sample?.title || p.itemType;
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-3">
        <div className="flex-1 relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} />
          <input value={search} onChange={e => setSearch(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') { setQ(search); setPage(1); } }}
            placeholder="Search buyer, artist…"
            className="w-full pl-9 pr-4 py-2.5 rounded-xl text-sm outline-none"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }} />
        </div>
        <button onClick={() => { setQ(search); setPage(1); }}
          className="px-5 py-2.5 rounded-xl text-sm font-bold"
          style={{ background: 'var(--green)', color: '#0a0a0a' }}>
          Search
        </button>
      </div>

      {data && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {data.total} confirmed sales · page {data.page}/{data.pages || 1}
        </p>
      )}

      <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" style={{ minWidth: 900 }}>
            <thead>
              <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                {['Date', 'Buyer', 'Artist', 'Type', 'Item', 'License', 'Gross', 'Vuka Fee', 'Artist Net', 'Status', 'Ref'].map(h => (
                  <th key={h} className="px-3 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={11} className="px-4 py-10 text-center">
                  <Loader2 className="animate-spin mx-auto" size={20} style={{ color: 'var(--green)' }} />
                </td></tr>
              ) : !data?.purchases?.length ? (
                <tr><td colSpan={11} className="px-4 py-10 text-center" style={{ color: 'var(--text-muted)' }}>No confirmed sales found</td></tr>
              ) : data.purchases.map((p: any) => (
                <tr key={p.id} className="border-t hover:bg-white/[0.02]" style={{ borderColor: 'var(--border)' }}>
                  <td className="px-3 py-3 whitespace-nowrap font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
                    {new Date(p.createdAt).toLocaleDateString('en-ZA')}
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap max-w-[140px]">
                    <div className="truncate font-medium">{p.buyerName || '—'}</div>
                    <div className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>{p.buyerEmail}</div>
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap font-medium" style={{ color: 'var(--sky)' }}>
                    {p.artist?.name || '—'}
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap">
                    <Pill label={p.itemType} />
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap max-w-[140px]">
                    <div className="truncate" style={{ color: 'var(--text-muted)' }}>{getItemTitle(p)}</div>
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap text-xs" style={{ color: 'var(--text-muted)' }}>
                    {p.licenseType || '—'}
                  </td>
                  <td className="px-3 py-3 font-mono font-bold whitespace-nowrap" style={{ color: 'var(--green)' }}>
                    {formatCurrency(p.amount)}
                  </td>
                  <td className="px-3 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--gold)' }}>
                    {formatCurrency(p.platformFee)}
                  </td>
                  <td className="px-3 py-3 font-mono whitespace-nowrap" style={{ color: '#38b6e8' }}>
                    {formatCurrency(p.netAmount)}
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap">
                    <Pill label={p.status} />
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
                    {(p.paystackReference || p.stripePaymentId || '').slice(0, 14) || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {data && data.pages > 1 && (
        <div className="flex gap-2 justify-end">
          {[...Array(data.pages)].map((_, i) => (
            <button key={i} onClick={() => setPage(i + 1)}
              className="w-8 h-8 rounded-lg text-sm font-mono"
              style={{
                background: page === i + 1 ? 'var(--green)' : 'var(--surface)',
                color: page === i + 1 ? '#0a0a0a' : 'var(--text)',
                border: '1px solid var(--border)',
              }}>
              {i + 1}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Tips tab ──────────────────────────────────────────────────────────────────
function TipsTab() {
  const [data, setData]       = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage]       = useState(1);

  const load = useCallback(() => {
    setLoading(true);
    fetch(`/api/admin/finance?view=tips&page=${page}`)
      .then(r => r.json()).then(setData).finally(() => setLoading(false));
  }, [page]);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="space-y-4">
      {data && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {data.total} confirmed tips/support payments
        </p>
      )}
      <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" style={{ minWidth: 700 }}>
            <thead>
              <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                {['Date', 'Fan', 'Artist', 'Tier', 'Gross', 'Vuka Fee', 'Artist Net', 'Message'].map(h => (
                  <th key={h} className="px-3 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={8} className="px-4 py-10 text-center">
                  <Loader2 className="animate-spin mx-auto" size={20} style={{ color: 'var(--green)' }} />
                </td></tr>
              ) : !data?.tips?.length ? (
                <tr><td colSpan={8} className="px-4 py-10 text-center" style={{ color: 'var(--text-muted)' }}>No tips found yet</td></tr>
              ) : data.tips.map((t: any) => (
                <tr key={t.id} className="border-t hover:bg-white/[0.02]" style={{ borderColor: 'var(--border)' }}>
                  <td className="px-3 py-3 font-mono text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                    {new Date(t.createdAt).toLocaleDateString('en-ZA')}
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap">
                    <div className="font-medium">{t.fanName || '—'}</div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{t.fanEmail}</div>
                  </td>
                  <td className="px-3 py-3 font-medium whitespace-nowrap" style={{ color: 'var(--sky)' }}>
                    {t.artist?.name || '—'}
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap">
                    <Pill label={t.tier || 'Listener'} color="#a0e87c" />
                  </td>
                  <td className="px-3 py-3 font-mono font-bold whitespace-nowrap" style={{ color: 'var(--green)' }}>
                    {formatCurrency(t.amount)}
                  </td>
                  <td className="px-3 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--gold)' }}>
                    {formatCurrency(t.platformFee)}
                  </td>
                  <td className="px-3 py-3 font-mono whitespace-nowrap" style={{ color: '#38b6e8' }}>
                    {formatCurrency(t.netAmount)}
                  </td>
                  <td className="px-3 py-3 max-w-[180px]">
                    <p className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>{t.message || '—'}</p>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ─── Subscriptions tab ────────────────────────────────────────────────────────
function SubscriptionsTab() {
  const [data, setData]   = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage]   = useState(1);

  const load = useCallback(() => {
    setLoading(true);
    fetch(`/api/admin/finance?view=purchases&page=${page}&itemType=subscription`)
      .then(r => r.json()).then(setData).finally(() => setLoading(false));
  }, [page]);

  useEffect(() => { load(); }, [load]);

  const PLAN_LABELS: Record<string, string> = {
    pro:   'Pro — R249/mo',
    label: 'Label — R999/mo',
  };

  return (
    <div className="space-y-4">
      {data && (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {data.total} confirmed plan payments · page {data.page}/{data.pages || 1}
        </p>
      )}
      <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" style={{ minWidth: 700 }}>
            <thead>
              <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                {['Date', 'Artist', 'Plan', 'Amount', 'Vuka Keeps', 'Ref'].map(h => (
                  <th key={h} className="px-4 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={6} className="px-4 py-10 text-center">
                  <Loader2 className="animate-spin mx-auto" size={20} style={{ color: 'var(--green)' }} />
                </td></tr>
              ) : !data?.purchases?.length ? (
                <tr><td colSpan={6} className="px-4 py-10 text-center" style={{ color: 'var(--text-muted)' }}>No plan subscriptions yet</td></tr>
              ) : data.purchases.map((p: any) => (
                <tr key={p.id} className="border-t hover:bg-white/[0.02]" style={{ borderColor: 'var(--border)' }}>
                  <td className="px-4 py-3 font-mono text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                    {new Date(p.createdAt).toLocaleDateString('en-ZA')}
                  </td>
                  <td className="px-4 py-3 font-medium whitespace-nowrap" style={{ color: 'var(--sky)' }}>
                    {p.artist?.name || p.buyerName || '—'}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <span className="px-2 py-0.5 rounded-full text-[11px] font-bold" style={{ background: '#c084fc22', color: '#c084fc' }}>
                      {PLAN_LABELS[p.licenseType] || p.licenseType || 'Platform Plan'}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono font-bold whitespace-nowrap" style={{ color: 'var(--green)' }}>
                    {formatCurrency(p.amount)}
                  </td>
                  <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--gold)' }}>
                    {formatCurrency(p.platformFee)}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                    {(p.paystackReference || '').slice(0, 14) || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {(data?.pages ?? 0) > 1 && (
        <div className="flex gap-2 justify-center flex-wrap">
          {Array.from({ length: data.pages }, (_: any, i: number) => (
            <button key={i} onClick={() => setPage(i + 1)}
              className="w-9 h-9 rounded-xl text-sm font-mono"
              style={{
                background: page === i + 1 ? 'var(--green)' : 'var(--surface)',
                color:      page === i + 1 ? '#0a0a0a' : 'var(--text)',
                border:     '1px solid var(--border)',
              }}>{i + 1}</button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Per-artist detail ─────────────────────────────────────────────────────────
function ArtistDetail({ artistId, onBack }: { artistId: string; onBack: () => void }) {
  const [data, setData]           = useState<any>(null);
  const [loading, setLoading]     = useState(true);
  const [subTab, setSubTab]       = useState<'sales' | 'tips' | 'payouts'>('sales');
  const [payModal, setPayModal]   = useState(false);
  const [selBank, setSelBank]     = useState<string>('');
  const [payAmt, setPayAmt]       = useState('');
  const [payRef, setPayRef]       = useState('');
  const [payNote, setPayNote]     = useState('');
  const [working, setWorking]     = useState(false);

  const load = () => {
    setLoading(true);
    fetch(`/api/admin/finance?view=artist&id=${artistId}`)
      .then(async r => {
        const d = await r.json();
        // Preserve error shape so the error UI can show it
        if (!r.ok && !d.error) d.error = `Server error (${r.status})`;
        setData(d);
      })
      .catch(() => setData({ error: 'Network error — could not reach server' }))
      .finally(() => setLoading(false));
  };

  useEffect(load, [artistId]);

  async function sendPayment() {
    if (!payAmt || parseFloat(payAmt) <= 0) return alert('Enter a valid amount');
    setWorking(true);
    try {
      // Step 1: create approved payout request
      const r1 = await fetch('/api/admin/finance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'create_payout',
          artistId,
          amount: parseFloat(payAmt),
          bankAccountId: selBank || undefined,
          notes: payNote || undefined,
        }),
      });
      const d1 = await r1.json();
      if (!r1.ok) return alert(d1.error || 'Failed to create payout');

      // Step 2: immediately mark paid
      const r2 = await fetch('/api/admin/finance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'mark_paid',
          requestId: d1.request.id,
          reference: payRef || undefined,
          notes: payNote || undefined,
        }),
      });
      const d2 = await r2.json();
      if (!r2.ok) return alert(d2.error || 'Failed to mark paid');

      setPayModal(false); setPayAmt(''); setPayRef(''); setPayNote(''); setSelBank('');
      load();
    } finally { setWorking(false); }
  }

  if (loading) return <div className="flex items-center justify-center py-20"><Loader2 className="animate-spin" size={24} style={{ color: 'var(--green)' }} /></div>;
  if (!data) return null;

  // API returned an error object (e.g. 503 DB error, 404 not found)
  if (data.error || !data.summary) {
    return (
      <div className="space-y-4">
        <button onClick={onBack}
          className="flex items-center gap-1.5 text-sm px-3 py-2 rounded-xl"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
          <ArrowLeft size={13} /> Back
        </button>
        <div className="flex items-center gap-3 p-5 rounded-2xl"
          style={{ background: 'var(--surface)', border: '1px solid rgba(255,77,77,0.25)' }}>
          <AlertTriangle size={18} style={{ color: '#ff4d4d', flexShrink: 0 }} />
          <div>
            <p className="font-bold text-sm" style={{ color: '#ff4d4d' }}>Failed to load artist data</p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
              {data.error || 'Unexpected response from server. Try refreshing.'}
            </p>
          </div>
          <button onClick={load} className="ml-auto text-xs px-3 py-1.5 rounded-lg font-semibold"
            style={{ background: 'var(--surface2)', color: 'var(--text)' }}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  const { artist, summary, purchases, tips, payoutRequests, payoutsLedger, bankAccounts = [] } = data;
  const defaultBank = bankAccounts.find((b: any) => b.isDefault) || bankAccounts[0] || null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <button onClick={onBack}
            className="flex items-center gap-1.5 text-sm px-3 py-2 rounded-xl"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
            <ArrowLeft size={13} /> Back
          </button>
          {artist?.photoUrl
            ? <img src={artist.photoUrl} className="w-9 h-9 rounded-full object-cover" />
            : <div className="w-9 h-9 rounded-full" style={{ background: 'var(--surface2)' }} />}
          <div>
            <p className="font-black text-lg">{artist?.name}</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Full financial breakdown</p>
          </div>
        </div>
        {summary.balance > 0 && (
          <button onClick={() => { setPayAmt(summary.balance.toFixed(2)); setSelBank(defaultBank?.id || ''); setPayModal(true); }}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold"
            style={{ background: 'var(--green)', color: '#0a0a0a' }}>
            <DollarSign size={14} /> Send Payment · {formatCurrency(summary.balance)}
          </button>
        )}
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Total Gross" value={formatCurrency(summary.grossSales + summary.grossTips)} color="var(--green)" sub={`${summary.salesCount} sales · ${summary.tipsCount} tips`} />
        <StatCard label="Vuka Keeps" value={formatCurrency(summary.totalPlatform)} color="var(--gold)" sub="per artist plan rate" />
        <StatCard label="Artist Total" value={formatCurrency(summary.totalEarned)} color="#38b6e8" sub="after platform fee" />
        <StatCard label="Balance Owed" value={formatCurrency(summary.balance)} color={summary.balance > 0 ? '#e8c87c' : '#a0e87c'} sub={`Paid out: ${formatCurrency(summary.paidOut)}`} />
      </div>

      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Sales Gross" value={formatCurrency(summary.grossSales)} sub={`${summary.salesCount} transactions`} />
        <StatCard label="Tips Gross" value={formatCurrency(summary.grossTips)} sub={`${summary.tipsCount} tips`} />
        <StatCard label="Payout Requests" value={String(payoutRequests.length)} sub={`${payoutRequests.filter((p: any) => p.status === 'pending').length} pending`} />
      </div>

      {/* Bank accounts panel */}
      {bankAccounts.length > 0 ? (
        <div className="rounded-2xl p-4 space-y-3" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Saved Bank Accounts</p>
          <div className="grid gap-2">
            {bankAccounts.map((b: any) => (
              <div key={b.id} className="flex items-center justify-between gap-3 px-4 py-3 rounded-xl"
                style={{ background: 'var(--bg)', border: b.isDefault ? '1px solid rgba(160,232,124,0.35)' : '1px solid var(--border)' }}>
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: b.isDefault ? 'rgba(160,232,124,0.15)' : 'var(--surface2)' }}>
                    <DollarSign size={14} style={{ color: b.isDefault ? 'var(--green)' : 'var(--text-muted)' }} />
                  </div>
                  <div className="min-w-0">
                    <div className="font-semibold text-sm">{b.bankName} <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{b.maskedNumber}</span></div>
                    <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      {b.accountHolder} · {b.accountType} · Branch: {b.branchCode || '—'}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {b.isDefault && <span className="text-[10px] font-bold px-2 py-0.5 rounded-full" style={{ background: 'rgba(160,232,124,0.15)', color: 'var(--green)' }}>DEFAULT</span>}
                  {summary.balance > 0 && (
                    <button onClick={() => { setSelBank(b.id); setPayAmt(summary.balance.toFixed(2)); setPayModal(true); }}
                      className="text-xs px-3 py-1.5 rounded-lg font-semibold"
                      style={{ background: 'rgba(160,232,124,0.1)', color: 'var(--green)', border: '1px solid rgba(160,232,124,0.2)' }}>
                      Pay to this
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="px-4 py-3 rounded-xl text-sm" style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
          No bank accounts saved by this artist yet.
        </div>
      )}

      {/* Sub tabs */}
      <div className="flex gap-2">
        {(['sales', 'tips', 'payouts'] as const).map(t => (
          <button key={t} onClick={() => setSubTab(t)}
            className="px-4 py-2 rounded-xl text-sm font-medium capitalize"
            style={{
              background: subTab === t ? 'rgba(160,232,124,0.12)' : 'var(--surface)',
              color: subTab === t ? 'var(--green)' : 'var(--text-muted)',
              border: subTab === t ? '1px solid rgba(160,232,124,0.3)' : '1px solid var(--border)',
            }}>
            {t} {t === 'sales' ? `(${purchases.length})` : t === 'tips' ? `(${tips.length})` : `(${payoutRequests.length})`}
          </button>
        ))}
      </div>

      {/* Sales table */}
      {subTab === 'sales' && (
        <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm" style={{ minWidth: 800 }}>
              <thead>
                <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                  {['Date', 'Buyer', 'Type', 'Item', 'License', 'Gross', 'Vuka', 'Artist', 'Status'].map(h => (
                    <th key={h} className="px-3 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {!purchases.length ? (
                  <tr><td colSpan={9} className="px-4 py-8 text-center" style={{ color: 'var(--text-muted)' }}>No sales yet</td></tr>
                ) : purchases.map((p: any) => (
                  <tr key={p.id} className="border-t hover:bg-white/[0.02]" style={{ borderColor: 'var(--border)' }}>
                    <td className="px-3 py-3 font-mono text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                      {new Date(p.createdAt).toLocaleDateString('en-ZA')}
                    </td>
                    <td className="px-3 py-3 whitespace-nowrap max-w-[130px]">
                      <div className="truncate text-sm font-medium">{p.buyerName}</div>
                      <div className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>{p.buyerEmail}</div>
                    </td>
                    <td className="px-3 py-3 whitespace-nowrap"><Pill label={p.itemType} /></td>
                    <td className="px-3 py-3 whitespace-nowrap max-w-[130px]">
                      <div className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>
                        {p.beat?.title || p.release?.title || p.video?.title || p.sample?.title || '—'}
                      </div>
                    </td>
                    <td className="px-3 py-3 text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{p.licenseType || '—'}</td>
                    <td className="px-3 py-3 font-mono font-bold whitespace-nowrap" style={{ color: 'var(--green)' }}>{formatCurrency(p.amount)}</td>
                    <td className="px-3 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--gold)' }}>{formatCurrency(p.platformFee)}</td>
                    <td className="px-3 py-3 font-mono whitespace-nowrap" style={{ color: '#38b6e8' }}>{formatCurrency(p.netAmount)}</td>
                    <td className="px-3 py-3 whitespace-nowrap"><Pill label={p.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tips table */}
      {subTab === 'tips' && (
        <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm" style={{ minWidth: 600 }}>
              <thead>
                <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                  {['Date', 'Fan', 'Tier', 'Gross', 'Vuka', 'Artist', 'Message'].map(h => (
                    <th key={h} className="px-3 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {!tips.length ? (
                  <tr><td colSpan={7} className="px-4 py-8 text-center" style={{ color: 'var(--text-muted)' }}>No tips yet</td></tr>
                ) : tips.map((t: any) => (
                  <tr key={t.id} className="border-t hover:bg-white/[0.02]" style={{ borderColor: 'var(--border)' }}>
                    <td className="px-3 py-3 font-mono text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{new Date(t.createdAt).toLocaleDateString('en-ZA')}</td>
                    <td className="px-3 py-3 whitespace-nowrap">
                      <div className="font-medium">{t.fanName}</div>
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{t.fanEmail}</div>
                    </td>
                    <td className="px-3 py-3 whitespace-nowrap"><Pill label={t.tier || 'Listener'} color="#a0e87c" /></td>
                    <td className="px-3 py-3 font-mono font-bold whitespace-nowrap" style={{ color: 'var(--green)' }}>{formatCurrency(t.amount)}</td>
                    <td className="px-3 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--gold)' }}>{formatCurrency(t.platformFee)}</td>
                    <td className="px-3 py-3 font-mono whitespace-nowrap" style={{ color: '#38b6e8' }}>{formatCurrency(t.netAmount)}</td>
                    <td className="px-3 py-3 max-w-[160px]"><p className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>{t.message || '—'}</p></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Payouts table */}
      {subTab === 'payouts' && (
        <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm" style={{ minWidth: 600 }}>
              <thead>
                <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                  {['Date', 'Amount', 'Method', 'Bank', 'Status', 'Notes'].map(h => (
                    <th key={h} className="px-3 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {!payoutRequests.length ? (
                  <tr><td colSpan={6} className="px-4 py-8 text-center" style={{ color: 'var(--text-muted)' }}>No payout requests</td></tr>
                ) : payoutRequests.map((p: any) => (
                  <tr key={p.id} className="border-t" style={{ borderColor: 'var(--border)' }}>
                    <td className="px-3 py-3 font-mono text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{new Date(p.createdAt).toLocaleDateString('en-ZA')}</td>
                    <td className="px-3 py-3 font-mono font-bold whitespace-nowrap" style={{ color: 'var(--green)' }}>{formatCurrency(p.amount)}</td>
                    <td className="px-3 py-3 whitespace-nowrap capitalize" style={{ color: 'var(--text-muted)' }}>{p.bankAccount ? 'bank' : 'paystack'}</td>
                    <td className="px-3 py-3 text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                      {p.bankAccount ? `${p.bankAccount.bankName} ····${p.bankAccount.maskedNumber?.slice(-4)}` : '—'}
                    </td>
                    <td className="px-3 py-3 whitespace-nowrap"><Pill label={p.status} /></td>
                    <td className="px-3 py-3 max-w-[160px]"><p className="truncate text-xs" style={{ color: 'var(--text-muted)' }}>{p.adminNotes || '—'}</p></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Send Payment modal */}
      {payModal && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={() => setPayModal(false)}>
          <div className="w-full max-w-md rounded-2xl p-6 space-y-4"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
            onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <p className="font-black text-base">Send Payment to {artist?.name}</p>
              <button onClick={() => setPayModal(false)} style={{ color: 'var(--text-muted)' }}>✕</button>
            </div>

            {/* Bank account selector */}
            {bankAccounts.length > 0 ? (
              <div className="space-y-2">
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Pay to account</p>
                {bankAccounts.map((b: any) => (
                  <button key={b.id} onClick={() => setSelBank(b.id)}
                    className="w-full flex items-center justify-between gap-3 px-4 py-3 rounded-xl text-left"
                    style={{
                      background: selBank === b.id ? 'rgba(160,232,124,0.1)' : 'var(--bg)',
                      border: selBank === b.id ? '1px solid rgba(160,232,124,0.4)' : '1px solid var(--border)',
                    }}>
                    <div>
                      <div className="text-sm font-semibold">{b.bankName} <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{b.maskedNumber}</span></div>
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{b.accountHolder} · {b.accountType} · Branch {b.branchCode || '—'}</div>
                    </div>
                    {b.isDefault && <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full flex-shrink-0" style={{ background: 'rgba(160,232,124,0.15)', color: 'var(--green)' }}>DEFAULT</span>}
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-sm px-3 py-2 rounded-xl" style={{ background: 'var(--bg)', color: '#e8a87c' }}>
                No bank accounts saved — payment will be recorded without a bank destination.
              </p>
            )}

            {/* Amount */}
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest mb-1" style={{ color: 'var(--text-muted)' }}>Amount (ZAR)</p>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 font-mono text-sm" style={{ color: 'var(--text-muted)' }}>R</span>
                <input type="number" value={payAmt} onChange={e => setPayAmt(e.target.value)} step="0.01" min="1"
                  className="w-full pl-7 pr-4 py-2.5 rounded-xl text-sm font-mono outline-none"
                  style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />
              </div>
              {summary.balance > 0 && (
                <button onClick={() => setPayAmt(summary.balance.toFixed(2))}
                  className="text-xs mt-1" style={{ color: 'var(--green)' }}>
                  Fill full balance: {formatCurrency(summary.balance)}
                </button>
              )}
            </div>

            {/* EFT reference */}
            <input value={payRef} onChange={e => setPayRef(e.target.value)}
              placeholder="EFT reference / bank ref…"
              className="w-full px-3 py-2.5 rounded-xl text-sm outline-none"
              style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />

            {/* Note */}
            <textarea value={payNote} onChange={e => setPayNote(e.target.value)}
              placeholder="Internal note (optional)…" rows={2}
              className="w-full px-3 py-2 rounded-xl text-sm resize-none outline-none"
              style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />

            <button onClick={sendPayment} disabled={working || !payAmt}
              className="w-full py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2"
              style={{ background: working || !payAmt ? 'var(--surface2)' : 'var(--green)', color: '#0a0a0a', opacity: working ? 0.7 : 1 }}>
              {working ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle size={14} />}
              {working ? 'Processing…' : `Mark Paid · R${parseFloat(payAmt || '0').toFixed(2)}`}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Artists tab ───────────────────────────────────────────────────────────────
function ArtistsTab({ onArtistClick }: { onArtistClick: (id: string) => void }) {
  const [data, setData]           = useState<any>(null);
  const [loading, setLoading]     = useState(true);
  const [quickPay, setQuickPay]   = useState<any>(null); // { artistId, artistName, balance, defaultBank }
  const [payAmt, setPayAmt]       = useState('');
  const [payRef, setPayRef]       = useState('');
  const [working, setWorking]     = useState(false);

  const load = () => {
    setLoading(true);
    fetch('/api/admin/finance?view=overview')
      .then(r => r.json()).then(setData).finally(() => setLoading(false));
  };

  useEffect(load, []);

  async function sendQuickPay() {
    if (!payAmt || parseFloat(payAmt) <= 0) return alert('Enter a valid amount');
    setWorking(true);
    try {
      const r1 = await fetch('/api/admin/finance', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'create_payout', artistId: quickPay.artistId, amount: parseFloat(payAmt), bankAccountId: quickPay.defaultBank?.id || undefined }),
      });
      const d1 = await r1.json();
      if (!r1.ok) return alert(d1.error || 'Failed');
      const r2 = await fetch('/api/admin/finance', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'mark_paid', requestId: d1.request.id, reference: payRef || undefined }),
      });
      if (!r2.ok) { const d2 = await r2.json(); return alert(d2.error || 'Failed'); }
      setQuickPay(null); setPayAmt(''); setPayRef(''); load();
    } finally { setWorking(false); }
  }

  if (loading) return <div className="flex items-center justify-center py-20"><Loader2 className="animate-spin" size={24} style={{ color: 'var(--green)' }} /></div>;

  const artists = data?.topArtists || [];

  return (
    <div className="space-y-4">
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Click a row to see full breakdown · Click <strong>Pay</strong> to send an EFT payment</p>
      <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" style={{ minWidth: 900 }}>
            <thead>
              <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                {['Artist', 'Sales', 'Tips', 'Artist Gets', 'Paid Out', 'Balance Owed', 'Bank Account', ''].map(h => (
                  <th key={h} className="px-4 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {!artists.length ? (
                <tr><td colSpan={8} className="px-4 py-10 text-center" style={{ color: 'var(--text-muted)' }}>No sales data yet</td></tr>
              ) : artists.map((row: any) => {
                const balance = row.artistOwes - row.payoutsTotal;
                const bank = row.defaultBank;
                return (
                  <tr key={row.artistId}
                    onClick={() => onArtistClick(row.artistId)}
                    className="border-t hover:bg-white/[0.02] cursor-pointer"
                    style={{ borderColor: 'var(--border)' }}>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        {row.artist?.photoUrl
                          ? <img src={row.artist.photoUrl} className="w-7 h-7 rounded-full object-cover flex-shrink-0" />
                          : <div className="w-7 h-7 rounded-full flex-shrink-0" style={{ background: 'var(--surface2)' }} />}
                        <span className="font-semibold">{row.artist?.name || '—'}</span>
                        <ChevronRight size={12} style={{ color: 'var(--text-muted)' }} />
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{row.salesCount}</td>
                    <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{row.tipsCount}</td>
                    <td className="px-4 py-3 font-mono font-bold whitespace-nowrap" style={{ color: '#38b6e8' }}>{formatCurrency(row.artistOwes)}</td>
                    <td className="px-4 py-3 font-mono whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{formatCurrency(row.payoutsTotal)}</td>
                    <td className="px-4 py-3 font-mono font-bold whitespace-nowrap"
                      style={{ color: balance > 0.01 ? '#e8c87c' : '#a0e87c' }}>
                      {formatCurrency(balance)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      {bank ? (
                        <div>
                          <div className="text-xs font-semibold">{bank.bankName} <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{bank.maskedNumber}</span></div>
                          <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{bank.accountHolder} · Branch {bank.branchCode || '—'}</div>
                        </div>
                      ) : <span className="text-xs" style={{ color: 'var(--text-muted)' }}>No bank saved</span>}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap" onClick={e => e.stopPropagation()}>
                      {balance > 0.01 && (
                        <button onClick={() => { setQuickPay({ artistId: row.artistId, artistName: row.artist?.name, balance, defaultBank: bank }); setPayAmt(balance.toFixed(2)); setPayRef(''); }}
                          className="px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1"
                          style={{ background: 'var(--green)', color: '#0a0a0a' }}>
                          <DollarSign size={11} /> Pay
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick-pay modal */}
      {quickPay && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={() => setQuickPay(null)}>
          <div className="w-full max-w-sm rounded-2xl p-6 space-y-4"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
            onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <p className="font-black">Pay {quickPay.artistName}</p>
              <button onClick={() => setQuickPay(null)} style={{ color: 'var(--text-muted)' }}>✕</button>
            </div>

            {quickPay.defaultBank ? (
              <div className="px-4 py-3 rounded-xl text-sm" style={{ background: 'var(--bg)', border: '1px solid rgba(160,232,124,0.25)' }}>
                <div className="font-semibold">{quickPay.defaultBank.bankName} <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{quickPay.defaultBank.maskedNumber}</span></div>
                <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{quickPay.defaultBank.accountHolder} · {quickPay.defaultBank.accountType} · Branch: {quickPay.defaultBank.branchCode || '—'}</div>
              </div>
            ) : (
              <p className="text-xs px-3 py-2 rounded-xl" style={{ background: 'var(--bg)', color: '#e8a87c' }}>No bank account saved — payment will be logged without a bank destination.</p>
            )}

            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 font-mono text-sm" style={{ color: 'var(--text-muted)' }}>R</span>
              <input type="number" value={payAmt} onChange={e => setPayAmt(e.target.value)} step="0.01" min="1"
                className="w-full pl-7 pr-4 py-2.5 rounded-xl text-sm font-mono outline-none"
                style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />
            </div>

            <input value={payRef} onChange={e => setPayRef(e.target.value)}
              placeholder="EFT reference / bank ref…"
              className="w-full px-3 py-2.5 rounded-xl text-sm outline-none"
              style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />

            <button onClick={sendQuickPay} disabled={working || !payAmt}
              className="w-full py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2"
              style={{ background: working || !payAmt ? 'var(--surface2)' : 'var(--green)', color: '#0a0a0a' }}>
              {working ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle size={14} />}
              {working ? 'Processing…' : `Mark Paid · R${parseFloat(payAmt || '0').toFixed(2)}`}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Payouts tab ───────────────────────────────────────────────────────────────
function PayoutsTab() {
  const [data, setData]         = useState<any>(null);
  const [loading, setLoading]   = useState(true);
  const [status, setStatus]     = useState('all');
  const [selected, setSelected] = useState<any>(null);
  const [working, setWorking]   = useState(false);
  const [refNum, setRefNum]     = useState('');
  const [note, setNote]         = useState('');

  const load = useCallback(() => {
    setLoading(true);
    fetch(`/api/admin/finance?view=payouts&status=${status}`)
      .then(r => r.json()).then(setData).finally(() => setLoading(false));
  }, [status]);

  useEffect(() => { load(); }, [load]);

  async function act(payoutId: string, action: 'approve' | 'reject' | 'mark_paid') {
    setWorking(true);
    const res = await fetch('/api/admin/finance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, requestId: payoutId, notes: note, reference: refNum }),
    });
    if (res.ok) { setSelected(null); setRefNum(''); setNote(''); load(); }
    else { const d = await res.json(); alert(d.error || 'Action failed'); }
    setWorking(false);
  }

  return (
    <div className="space-y-4">
      {/* Status filter */}
      <div className="flex gap-2 flex-wrap">
        {['all', 'pending', 'approved', 'paid', 'rejected'].map(s => (
          <button key={s} onClick={() => setStatus(s)}
            className="px-4 py-2 rounded-xl text-sm font-medium capitalize"
            style={{
              background: status === s ? 'rgba(160,232,124,0.12)' : 'var(--surface)',
              color: status === s ? 'var(--green)' : 'var(--text-muted)',
              border: status === s ? '1px solid rgba(160,232,124,0.3)' : '1px solid var(--border)',
            }}>
            {s}
          </button>
        ))}
      </div>

      <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm" style={{ minWidth: 800 }}>
            <thead>
              <tr style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
                {['Artist', 'Email', 'Amount', 'Bank', 'Account', 'Branch', 'Status', 'Requested', 'Actions'].map(h => (
                  <th key={h} className="px-3 py-3 text-left font-semibold whitespace-nowrap" style={{ color: 'var(--text-muted)', fontSize: 11 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={9} className="px-4 py-10 text-center">
                  <Loader2 className="animate-spin mx-auto" size={20} style={{ color: 'var(--green)' }} />
                </td></tr>
              ) : !data?.requests?.length ? (
                <tr><td colSpan={9} className="px-4 py-10 text-center" style={{ color: 'var(--text-muted)' }}>No payout requests</td></tr>
              ) : data.requests.map((p: any) => (
                <tr key={p.id} className="border-t hover:bg-white/[0.02]" style={{ borderColor: 'var(--border)' }}>
                  <td className="px-3 py-3 font-semibold whitespace-nowrap">{p.artist?.name || '—'}</td>
                  <td className="px-3 py-3 text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{p.artist?.user?.email || '—'}</td>
                  <td className="px-3 py-3 font-mono font-bold whitespace-nowrap" style={{ color: 'var(--green)' }}>{formatCurrency(p.amount)}</td>
                  <td className="px-3 py-3 whitespace-nowrap text-xs">{p.bankAccount?.bankName || '—'}</td>
                  <td className="px-3 py-3 whitespace-nowrap font-mono text-xs">{p.bankAccount?.maskedNumber || '—'}</td>
                  <td className="px-3 py-3 whitespace-nowrap font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{p.bankAccount?.branchCode || '—'}</td>
                  <td className="px-3 py-3 whitespace-nowrap"><Pill label={p.status} /></td>
                  <td className="px-3 py-3 font-mono text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                    {new Date(p.createdAt).toLocaleDateString('en-ZA')}
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap">
                    <button onClick={() => setSelected(p)}
                      className="px-3 py-1.5 rounded-lg text-xs font-semibold"
                      style={{ background: 'rgba(160,232,124,0.1)', color: 'var(--green)', border: '1px solid rgba(160,232,124,0.2)' }}>
                      Review
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Review modal */}
      {selected && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
          <div className="w-full max-w-md rounded-2xl p-6 space-y-4"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
            onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <p className="font-black text-base">Review Payout</p>
              <button onClick={() => setSelected(null)} style={{ color: 'var(--text-muted)' }}>✕</button>
            </div>

            <div className="p-4 rounded-xl space-y-2 text-sm" style={{ background: 'var(--bg)' }}>
              {[
                ['Artist', selected.artist?.name],
                ['Email',  selected.artist?.user?.email],
                ['Amount', formatCurrency(selected.amount)],
                ['Bank',   selected.bankAccount?.bankName],
                ['Account', selected.bankAccount?.maskedNumber],
                ['Branch', selected.bankAccount?.branchCode],
                ['Type',   selected.bankAccount?.accountType],
              ].filter(([,v]) => v).map(([k, v]) => (
                <div key={k as string} className="flex justify-between gap-4">
                  <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                  <span className="font-semibold text-right">{v}</span>
                </div>
              ))}
            </div>

            <input value={refNum} onChange={e => setRefNum(e.target.value)}
              placeholder="Bank reference / EFT ref (for approval/mark paid)…"
              className="w-full px-3 py-2.5 rounded-xl text-sm outline-none"
              style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />
            <textarea value={note} onChange={e => setNote(e.target.value)}
              placeholder="Note or rejection reason…" rows={2}
              className="w-full px-3 py-2 rounded-xl text-sm resize-none outline-none"
              style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }} />

            <div className="grid grid-cols-3 gap-2">
              {selected.status === 'pending' && (
                <button onClick={() => act(selected.id, 'approve')} disabled={working}
                  className="py-2.5 rounded-xl text-sm font-bold flex items-center justify-center gap-1.5"
                  style={{ background: 'rgba(160,232,124,0.15)', color: 'var(--green)', border: '1px solid rgba(160,232,124,0.3)' }}>
                  <CheckCircle size={13} /> Approve
                </button>
              )}
              {selected.status === 'approved' && (
                <button onClick={() => act(selected.id, 'mark_paid')} disabled={working}
                  className="py-2.5 rounded-xl text-sm font-bold flex items-center justify-center gap-1.5 col-span-2"
                  style={{ background: 'var(--green)', color: '#0a0a0a' }}>
                  <CheckCircle size={13} /> Mark Paid
                </button>
              )}
              {['pending', 'approved'].includes(selected.status) && (
                <button onClick={() => act(selected.id, 'reject')} disabled={working}
                  className="py-2.5 rounded-xl text-sm font-bold flex items-center justify-center gap-1.5"
                  style={{ background: 'rgba(255,77,77,0.1)', color: '#ff4d4d' }}>
                  <XCircle size={13} /> Reject
                </button>
              )}
              {!['pending', 'approved'].includes(selected.status) && (
                <button onClick={() => setSelected(null)}
                  className="col-span-3 py-2.5 rounded-xl text-sm font-semibold"
                  style={{ background: 'var(--surface2)', color: 'var(--text)' }}>
                  Close
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────
export default function AdminFinancePage() {
  const [tab, setTab]               = useState<MainTab>('overview');
  const [artistDrillId, setArtistDrillId] = useState<string | null>(null);

  function goArtist(id: string) {
    setArtistDrillId(id);
    setTab('artists');
  }

  const TABS: { id: MainTab; label: string }[] = [
    { id: 'overview',      label: 'Overview'       },
    { id: 'sales',         label: 'All Sales'       },
    { id: 'tips',          label: 'Tips'            },
    { id: 'subscriptions', label: 'Subscriptions'   },
    { id: 'artists',       label: 'Per Artist'      },
    { id: 'payouts',       label: 'Payouts'         },
  ];

  return (
    <div>
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black font-display">Finance</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Every sale, tip, and payout — fees vary by artist plan (Free 15% · Pro 8% · Label 5%)
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {TABS.map(t => (
          <button key={t.id} onClick={() => { setTab(t.id); if (t.id !== 'artists') setArtistDrillId(null); }}
            className="px-4 py-2 rounded-xl text-sm font-medium"
            style={{
              background: tab === t.id ? 'rgba(160,232,124,0.12)' : 'var(--surface)',
              color: tab === t.id ? 'var(--green)' : 'var(--text-muted)',
              border: tab === t.id ? '1px solid rgba(160,232,124,0.3)' : '1px solid var(--border)',
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab onArtistClick={goArtist} />}
      {tab === 'sales'    && <SalesTab />}
      {tab === 'tips'          && <TipsTab />}
      {tab === 'subscriptions' && <SubscriptionsTab />}
      {tab === 'artists'  && (
        artistDrillId
          ? <ArtistDetail artistId={artistDrillId} onBack={() => setArtistDrillId(null)} />
          : <ArtistsTab onArtistClick={goArtist} />
      )}
      {tab === 'payouts'  && <PayoutsTab />}
    </div>
  );
}
