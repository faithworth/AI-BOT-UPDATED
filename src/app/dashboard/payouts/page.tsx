'use client';
// src/app/dashboard/payouts/page.tsx
// FIXED: Removed Stripe. Added Ozow, Yoco, SA Bank EFT.
// FIXED: connected.paystack now reads directly from API (not crashed by Stripe import).

import { useEffect, useState } from 'react';
import { formatCurrency } from '@/lib/utils';
import {
  CheckCircle, Clock, TrendingUp, Wallet, ArrowUpRight,
  Loader2, RefreshCw, Building2, CreditCard, Zap, ExternalLink,
  Plus, Banknote,
} from 'lucide-react';

export default function PayoutsPage() {
  const [data, setData]     = useState<any>(null);
  const [artist, setArtist] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab]       = useState<'overview' | 'history'>('overview');

  // Bank account form
  const [showBankForm, setShowBankForm] = useState(false);
  const [bankSaving, setBankSaving]     = useState(false);
  const [bankForm, setBankForm]         = useState({
    accountHolder: '', bankName: '', branchCode: '', accountNumber: '',
  });
  const SA_BANKS = [
    { name: 'Absa', branch: '632005' },
    { name: 'Capitec', branch: '470010' },
    { name: 'FNB / First National Bank', branch: '250655' },
    { name: 'Nedbank', branch: '198765' },
    { name: 'Standard Bank', branch: '051001' },
    { name: 'African Bank', branch: '430000' },
    { name: 'Discovery Bank', branch: '679000' },
    { name: 'TymeBank', branch: '678910' },
    { name: 'Investec', branch: '580105' },
  ];

  async function load() {
    setLoading(true);
    const [payoutsRes, settingsRes] = await Promise.all([
      fetch('/api/dashboard/payouts'),
      fetch('/api/dashboard/settings'),
    ]);
    if (payoutsRes.ok)  setData(await payoutsRes.json());
    if (settingsRes.ok) setArtist((await settingsRes.json()).artist);
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  async function saveBankAccount() {
    if (!bankForm.accountHolder || !bankForm.bankName || !bankForm.accountNumber) return;
    setBankSaving(true);
    try {
      await fetch('/api/payouts/bank-accounts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          accountType: 'bank',
          accountHolder: bankForm.accountHolder,
          bankName: bankForm.bankName,
          branchCode: bankForm.branchCode,
          accountNumber: bankForm.accountNumber,
          isDefault: true,
        }),
      });
      setShowBankForm(false);
      setBankForm({ accountHolder: '', bankName: '', branchCode: '', accountNumber: '' });
      await load();
    } catch {}
    setBankSaving(false);
  }

  if (loading) return (
    <div className="p-10 flex items-center gap-3" style={{ color: 'var(--text-muted)' }}>
      <Loader2 size={18} className="animate-spin" /> Loading your earnings…
    </div>
  );

  const {
    payouts = [], summary = {}, connected = {},
    payoutRequests = [], bankAccounts = [],
  } = data || {};

  function statusBadge(status: string) {
    const map: Record<string, { label: string; color: string; bg: string }> = {
      pending:    { label: 'Pending',    color: 'var(--gold)',  bg: 'rgba(234,179,8,0.1)' },
      processing: { label: 'Processing', color: 'var(--sky)',   bg: 'rgba(56,182,232,0.1)' },
      completed:  { label: 'Paid',       color: 'var(--green)', bg: 'rgba(16,185,129,0.1)' },
      approved:   { label: 'Approved',   color: 'var(--green)', bg: 'rgba(16,185,129,0.1)' },
      failed:     { label: 'Failed',     color: '#f87171',      bg: 'rgba(248,113,113,0.1)' },
    };
    const s = map[status] || map.pending;
    return (
      <span className="text-xs font-semibold px-2 py-0.5 rounded-full"
        style={{ color: s.color, background: s.bg }}>
        {s.label}
      </span>
    );
  }

  return (
    <div className="p-6 md:p-10 max-w-3xl">
      <div className="flex items-center justify-between mb-2">
        <h1 className="text-2xl font-black" style={{ color: 'var(--text)' }}>Payouts</h1>
        <button onClick={load} className="p-2 rounded-lg" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
          <RefreshCw size={15} style={{ color: 'var(--text-muted)' }} />
        </button>
      </div>
      <p className="text-sm mb-8" style={{ color: 'var(--text-muted)' }}>
        Connect your payment accounts to receive your earnings from SA and African fans.
      </p>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-3 mb-8">
        {[
          { label: 'Total Earned', value: summary.totalEarned || 0, icon: TrendingUp, color: 'var(--green)' },
          { label: 'Paid Out',     value: summary.totalPaid    || 0, icon: CheckCircle, color: 'var(--sky)' },
          { label: 'Pending',      value: summary.totalPending || 0, icon: Clock, color: 'var(--gold)' },
        ].map(card => (
          <div key={card.label} className="p-4 rounded-2xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <card.icon size={16} style={{ color: card.color }} className="mb-2" />
            <div className="text-xl font-black" style={{ color: card.color }}>
              {formatCurrency(card.value)}
            </div>
            <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{card.label}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        {(['overview', 'history'] as const).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className="px-4 py-2 rounded-lg text-sm font-semibold capitalize"
            style={{
              background: tab === t ? 'var(--sky)' : 'var(--surface)',
              color: tab === t ? 'white' : 'var(--text-muted)',
              border: '1px solid var(--border)',
            }}>
            {t}
          </button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div className="space-y-4">

          {/* ── Paystack ── */}
          <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between px-6 py-4" style={{ background: 'var(--surface)' }}>
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'rgba(0,160,90,0.15)' }}>
                  <Wallet size={18} style={{ color: '#00a05a' }} />
                </div>
                <div>
                  <p className="font-bold text-sm" style={{ color: 'var(--text)' }}>Paystack</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>South African payments · ZAR · card, EFT, bank transfer</p>
                </div>
              </div>
              {connected.paystack
                ? <span className="text-xs font-semibold px-2 py-1 rounded-full" style={{ color: 'var(--green)', background: 'rgba(16,185,129,0.1)' }}>✓ Connected</span>
                : <span className="text-xs font-semibold px-2 py-1 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--surface2)' }}>Not connected</span>}
            </div>
            <div className="px-6 py-5" style={{ background: 'var(--surface2)', borderTop: '1px solid var(--border)' }}>
              {connected.paystack ? (
                <>
                  <div className="flex items-center gap-2 mb-3 p-3 rounded-xl"
                    style={{ background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.2)' }}>
                    <CheckCircle size={14} style={{ color: 'var(--green)' }} />
                    <p className="text-sm font-semibold" style={{ color: 'var(--green)' }}>
                      Bank account on file: {artist?.paystackRecipient}
                    </p>
                  </div>
                  <p className="text-sm mb-3" style={{ color: 'var(--text-muted)' }}>
                    Paystack collects payments on your behalf. Payouts are processed to your
                    SA bank account based on your payout requests below.
                  </p>
                  <a href="https://dashboard.paystack.com" target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-1.5 text-xs font-semibold"
                    style={{ color: '#00a05a' }}>
                    View Paystack Dashboard <ExternalLink size={11} />
                  </a>
                </>
              ) : (
                <>
                  <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                    Add your bank account so SA buyers can pay you via card, EFT, or bank transfer through Paystack.
                    Paystack collects payments on your behalf and you request payouts to this account.
                  </p>
                  <a href="/dashboard/settings" className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-sm text-white w-fit"
                    style={{ background: 'linear-gradient(135deg,#00a05a,#007a44)' }}>
                    <ArrowUpRight size={14} /> Add Bank Account in Settings
                  </a>
                  <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
                    Get your account details from your{' '}
                    <a href="https://dashboard.paystack.com/#/settings/business" target="_blank" rel="noopener noreferrer"
                      className="underline" style={{ color: 'var(--sky)' }}>
                      Paystack account settings
                    </a>.
                  </p>
                </>
              )}
            </div>
          </div>

          {/* ── Ozow ── */}
          <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between px-6 py-4" style={{ background: 'var(--surface)' }}>
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'rgba(80,80,220,0.12)' }}>
                  <Zap size={18} style={{ color: '#5050dc' }} />
                </div>
                <div>
                  <p className="font-bold text-sm" style={{ color: 'var(--text)' }}>Ozow</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Instant EFT · ZAR · No card needed · Bank-to-bank</p>
                </div>
              </div>
              <span className="text-xs font-semibold px-2 py-1 rounded-full" style={{ color: 'var(--sky)', background: 'rgba(56,182,232,0.1)' }}>
                Coming soon
              </span>
            </div>
            <div className="px-6 py-5" style={{ background: 'var(--surface2)', borderTop: '1px solid var(--border)' }}>
              <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                Ozow enables instant EFT payments directly from any South African bank — no card required.
                Ideal for fans who prefer banking apps over cards. Integration in progress.
              </p>
              <a href="https://ozow.com" target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-xs font-semibold"
                style={{ color: '#5050dc' }}>
                Learn about Ozow <ExternalLink size={11} />
              </a>
            </div>
          </div>

          {/* ── Yoco ── */}
          <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between px-6 py-4" style={{ background: 'var(--surface)' }}>
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'rgba(0,200,150,0.12)' }}>
                  <CreditCard size={18} style={{ color: '#00c896' }} />
                </div>
                <div>
                  <p className="font-bold text-sm" style={{ color: 'var(--text)' }}>Yoco</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>SA payment gateway · Cards, QR, online · ZAR</p>
                </div>
              </div>
              <span className="text-xs font-semibold px-2 py-1 rounded-full" style={{ color: 'var(--sky)', background: 'rgba(56,182,232,0.1)' }}>
                Coming soon
              </span>
            </div>
            <div className="px-6 py-5" style={{ background: 'var(--surface2)', borderTop: '1px solid var(--border)' }}>
              <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                Yoco is a South African payment gateway built for local businesses — accept Visa, Mastercard,
                and QR payments. Payouts go straight to your SA bank account within 1–2 business days.
              </p>
              <a href="https://www.yoco.com/za/" target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-xs font-semibold"
                style={{ color: '#00c896' }}>
                Learn about Yoco <ExternalLink size={11} />
              </a>
            </div>
          </div>

          {/* ── SA Bank EFT / Manual Payout ── */}
          <div className="rounded-2xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between px-6 py-4" style={{ background: 'var(--surface)' }}>
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'rgba(234,179,8,0.12)' }}>
                  <Building2 size={18} style={{ color: 'var(--gold)' }} />
                </div>
                <div>
                  <p className="font-bold text-sm" style={{ color: 'var(--text)' }}>SA Bank EFT</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Manual payout · FNB, Absa, Standard Bank, Capitec, Nedbank and more</p>
                </div>
              </div>
              {bankAccounts.length > 0
                ? <span className="text-xs font-semibold px-2 py-1 rounded-full" style={{ color: 'var(--green)', background: 'rgba(16,185,129,0.1)' }}>✓ Saved</span>
                : <span className="text-xs font-semibold px-2 py-1 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--surface2)' }}>Not set up</span>}
            </div>
            <div className="px-6 py-5" style={{ background: 'var(--surface2)', borderTop: '1px solid var(--border)' }}>
              {bankAccounts.length > 0 ? (
                <>
                  <div className="space-y-2 mb-4">
                    {bankAccounts.map((a: any) => (
                      <div key={a.id} className="flex items-center gap-3 p-3 rounded-xl"
                        style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                        <Building2 size={14} style={{ color: 'var(--gold)' }} />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>{a.bankName}</p>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            {a.accountHolder} · {a.maskedNumber || '****'}
                            {a.isDefault && <span className="ml-2 text-[10px] font-bold px-1.5 py-0.5 rounded" style={{ background: 'rgba(16,185,129,0.1)', color: 'var(--green)' }}>DEFAULT</span>}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-sm mb-3" style={{ color: 'var(--text-muted)' }}>
                    Manual EFT payouts are processed within 2–5 business days.
                    Request a payout from your History tab once your balance is confirmed.
                  </p>
                  <button onClick={() => setShowBankForm(v => !v)}
                    className="flex items-center gap-1.5 text-xs font-semibold"
                    style={{ color: 'var(--sky)' }}>
                    <Plus size={12} /> Add another account
                  </button>
                </>
              ) : (
                <>
                  <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                    Add your South African bank account details to receive manual EFT payouts.
                    Supports FNB, Absa, Standard Bank, Capitec, Nedbank, and all major SA banks.
                  </p>
                  <button onClick={() => setShowBankForm(v => !v)}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-sm text-white"
                    style={{ background: 'linear-gradient(135deg,#d4a000,#b38600)' }}>
                    <Banknote size={14} /> Add SA Bank Account
                  </button>
                </>
              )}

              {/* Bank account form */}
              {showBankForm && (
                <div className="mt-5 space-y-3 p-4 rounded-xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                  <p className="text-sm font-bold mb-3" style={{ color: 'var(--text)' }}>Add SA Bank Account</p>

                  <div>
                    <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Account Holder Name</label>
                    <input
                      value={bankForm.accountHolder}
                      onChange={e => setBankForm(p => ({ ...p, accountHolder: e.target.value }))}
                      placeholder="e.g. Tshepang Mokoena"
                      className="w-full px-3 py-2.5 rounded-lg text-sm"
                      style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                    />
                  </div>

                  <div>
                    <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Bank</label>
                    <select
                      value={bankForm.bankName}
                      onChange={e => {
                        const bank = SA_BANKS.find(b => b.name === e.target.value);
                        setBankForm(p => ({ ...p, bankName: e.target.value, branchCode: bank?.branch || '' }));
                      }}
                      className="w-full px-3 py-2.5 rounded-lg text-sm"
                      style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                    >
                      <option value="">Select your bank</option>
                      {SA_BANKS.map(b => (
                        <option key={b.name} value={b.name}>{b.name} (Branch: {b.branch})</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Branch Code</label>
                    <input
                      value={bankForm.branchCode}
                      onChange={e => setBankForm(p => ({ ...p, branchCode: e.target.value }))}
                      placeholder="Auto-filled from bank selection"
                      className="w-full px-3 py-2.5 rounded-lg text-sm"
                      style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                    />
                  </div>

                  <div>
                    <label className="block text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Account Number</label>
                    <input
                      value={bankForm.accountNumber}
                      onChange={e => setBankForm(p => ({ ...p, accountNumber: e.target.value }))}
                      placeholder="Your account number"
                      className="w-full px-3 py-2.5 rounded-lg text-sm"
                      style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text)' }}
                    />
                  </div>

                  <div className="flex gap-2 pt-1">
                    <button onClick={saveBankAccount} disabled={bankSaving}
                      className="flex-1 py-2.5 rounded-lg font-bold text-sm text-white disabled:opacity-60"
                      style={{ background: 'var(--sky)' }}>
                      {bankSaving ? <><Loader2 size={14} className="animate-spin inline mr-2" />Saving…</> : 'Save Account'}
                    </button>
                    <button onClick={() => setShowBankForm(false)}
                      className="px-4 py-2.5 rounded-lg text-sm font-medium"
                      style={{ background: 'var(--surface2)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* How it works */}
          <div className="p-5 rounded-2xl" style={{ background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.2)' }}>
            <p className="text-sm font-bold mb-3" style={{ color: 'var(--green)' }}>💚 How payouts work</p>
            <div className="space-y-2">
              {[
                'Paystack: once connected, request a payout and funds are sent to your SA bank account.',
                'SA Bank EFT: save your bank details, then request a payout when your balance is ready.',
                'Ozow and Yoco integrations are coming — they will appear here once live.',
                'All amounts are in ZAR. International buyers pay via card and funds convert automatically.',
              ].map((item, i) => (
                <p key={i} className="text-xs flex gap-2" style={{ color: 'var(--text-muted)' }}>
                  <span style={{ color: 'var(--green)' }}>✓</span> {item}
                </p>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── HISTORY TAB ── */}
      {tab === 'history' && (
        <div className="space-y-4">
          {/* Payout requests */}
          {payoutRequests.length > 0 && (
            <div>
              <p className="text-sm font-bold mb-3" style={{ color: 'var(--text)' }}>Payout Requests</p>
              <div className="space-y-2 mb-6">
                {payoutRequests.map((r: any) => (
                  <div key={r.id} className="flex items-center justify-between p-4 rounded-xl"
                    style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
                          {formatCurrency(r.amount)}
                        </p>
                        {statusBadge(r.status)}
                      </div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {new Date(r.createdAt).toLocaleDateString('en-ZA')}
                        {r.bankAccount && ` · ${r.bankAccount.bankName}`}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ArtistPayout records */}
          {payouts.length === 0 && payoutRequests.length === 0 ? (
            <div className="text-center py-16" style={{ color: 'var(--text-muted)' }}>
              <Wallet size={40} className="mx-auto mb-4 opacity-30" />
              <p className="font-semibold">No sales yet</p>
              <p className="text-sm mt-1">Your earnings will appear here after your first sale.</p>
            </div>
          ) : (
            payouts.length > 0 && (
              <div>
                <p className="text-sm font-bold mb-3" style={{ color: 'var(--text)' }}>Sales History</p>
                <div className="space-y-2">
                  {payouts.map((p: any) => (
                    <div key={p.id} className="flex items-center justify-between p-4 rounded-xl"
                      style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
                            {formatCurrency(p.amount)}
                          </p>
                          {statusBadge(p.status)}
                          <span className="text-xs px-1.5 py-0.5 rounded font-medium uppercase"
                            style={{ background: 'var(--surface2)', color: 'var(--text-muted)' }}>
                            {p.method}
                          </span>
                        </div>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          {new Date(p.createdAt).toLocaleDateString('en-ZA')}
                          {p.notes && ` · ${p.notes}`}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}
