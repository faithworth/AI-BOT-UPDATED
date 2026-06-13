'use client';
// src/app/services/page.tsx
// Unified public services hub — shows BOTH:
//   • Industry Professional services (management, promo, sync, legal…)
//   • Artist marketplace services (mixing, mastering, features, beats…)
//
// Industry services: artist pays → Vuka charges 10% fee on industry (auto via Paystack)
// Marketplace services: buyer pays → artist plan fee (5–15%) applies
//
// Both tabs are visible to everyone. Logged-in artists can buy/inquire/order.
// Industry users can browse artists via the "Find Artists" tab.

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Navbar from '@/components/Navbar';
import {
  Search, Loader2, Briefcase, Calendar, MessageSquare,
  CheckCircle, Send, ShoppingCart, Users, Star, Clock,
  Music, Mic2, Building2, ChevronRight, Zap, Shield,
  DollarSign, ArrowRight,
} from 'lucide-react';
import { createClient } from '@/lib/supabase';

// ─── Types ────────────────────────────────────────────────────────────────────
type ServiceTab = 'industry' | 'marketplace';

const INDUSTRY_CATS = [
  { value: '', label: 'All' },
  { value: 'promotion',    label: 'Promotion' },
  { value: 'distribution', label: 'Distribution' },
  { value: 'sync',         label: 'Sync & Licensing' },
  { value: 'management',   label: 'Management' },
  { value: 'scouting',     label: 'Talent Scouting' },
  { value: 'sponsorship',  label: 'Sponsorship' },
  { value: 'legal',        label: 'Legal' },
  { value: 'photography',  label: 'Photography' },
  { value: 'videography',  label: 'Videography' },
  { value: 'mixing',       label: 'Mixing' },
  { value: 'mastering',    label: 'Mastering' },
  { value: 'other',        label: 'Other' },
];

const MKTPLACE_CATS = [
  { value: '', label: 'All' },
  { value: 'mixing',       label: 'Mixing' },
  { value: 'mastering',    label: 'Mastering' },
  { value: 'features',     label: 'Features' },
  { value: 'production',   label: 'Production' },
  { value: 'ghostwriting', label: 'Ghostwriting' },
  { value: 'videography',  label: 'Videography' },
  { value: 'photography',  label: 'Photography' },
  { value: 'promotion',    label: 'Promotion' },
  { value: 'other',        label: 'Other' },
];

const PRICING_LABELS: Record<string, string> = {
  fixed: '', per_track: '/ track', per_month: '/ mo', quote: '(quote)',
};

// ─── Main Component ───────────────────────────────────────────────────────────
export default function ServicesPage() {
  const router = useRouter();
  const [tab, setTab]                   = useState<ServiceTab>('industry');
  const [search, setSearch]             = useState('');
  const [category, setCategory]         = useState('');
  const [sort, setSort]                 = useState('price_asc');

  // Industry
  const [industrySvcs, setIndustrySvcs] = useState<any[]>([]);
  const [indLoading, setIndLoading]     = useState(true);

  // Marketplace
  const [mktSvcs, setMktSvcs]           = useState<any[]>([]);
  const [mktLoading, setMktLoading]     = useState(false);
  const [mktLoaded, setMktLoaded]       = useState(false);

  // Auth
  const [userId, setUserId]             = useState<string | null>(null);
  const [userRole, setUserRole]         = useState<string | null>(null);

  // UI state
  const [inquiring, setInquiring]       = useState<string | null>(null);
  const [ordering, setOrdering]         = useState<string | null>(null);
  const [messaging, setMessaging]       = useState<string | null>(null);
  const [msgMap, setMsgMap]             = useState<Record<string, string>>({});
  const [showMsg, setShowMsg]           = useState<string | null>(null);
  const [doneMsgs, setDoneMsgs]         = useState<Record<string, string>>({});

  // Load auth
  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(async ({ data }) => {
      if (data.user) {
        const me = await fetch('/api/auth/me').then(r => r.ok ? r.json() : null);
        if (me) { setUserId(me.id); setUserRole(me.role || 'fan'); }
      }
    });
  }, []);

  // Load industry services
  useEffect(() => {
    if (tab !== 'industry') return;
    setIndLoading(true);
    const p = new URLSearchParams({ sort });
    if (category) p.set('category', category);
    fetch(`/api/industry/browse?${p}`)
      .then(r => r.json())
      .then(d => { setIndustrySvcs(d.services || []); setIndLoading(false); })
      .catch(() => setIndLoading(false));
  }, [tab, category, sort]);

  // Load marketplace (lazy)
  useEffect(() => {
    if (tab !== 'marketplace' || mktLoaded) return;
    setMktLoading(true);
    setMktLoaded(true);
    const p = new URLSearchParams({ take: '30' });
    if (category) p.set('category', category);
    fetch(`/api/marketplace/services?${p}`)
      .then(r => r.ok ? r.json() : { services: [] })
      .then(d => { setMktSvcs(d.services || []); setMktLoading(false); })
      .catch(() => setMktLoading(false));
  }, [tab, mktLoaded, category]);

  // Reset category when switching tabs
  function switchTab(t: ServiceTab) {
    setTab(t);
    setCategory('');
    setSearch('');
  }

  async function sendInquiry(serviceId: string) {
    if (!userId) { router.push('/auth/login'); return; }
    setInquiring(serviceId);
    const res = await fetch('/api/industry/inquire', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ serviceId, message: msgMap[serviceId] || '' }),
    });
    if (res.ok) {
      setDoneMsgs(d => ({ ...d, [serviceId]: 'inquiry' }));
      setShowMsg(null);
    } else {
      const err = await res.json().catch(() => ({}));
      alert(err.error || 'Failed to send inquiry');
    }
    setInquiring(null);
  }

  async function orderIndustryService(svc: any) {
    if (!userId) { router.push('/auth/login'); return; }
    setOrdering(svc.id);
    try {
      const res = await fetch('/api/industry/order', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ serviceId: svc.id, requirements: msgMap[svc.id] || '' }),
      });
      const data = await res.json();
      if (res.ok && data.payUrl) {
        // Redirect to Paystack — payment happens immediately
        window.location.href = data.payUrl;
      } else {
        alert(data.error || 'Failed to initiate payment');
        setOrdering(null);
      }
    } catch {
      alert('Network error. Please try again.');
      setOrdering(null);
    }
  }

  async function openMessage(svc: any) {
    const recipientId = svc.industryUser?.userId || svc.industryUser?.user?.id;
    if (!recipientId) return;
    setMessaging(svc.id);
    try {
      const res = await fetch('/api/messages/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ recipientId }),
      });
      if (res.ok) router.push('/messages');
    } catch {}
    setMessaging(null);
  }

  const cats = tab === 'industry' ? INDUSTRY_CATS : MKTPLACE_CATS;
  const canAct = userRole === 'artist' || userRole === 'fan' || userRole === 'producer';

  // Filter by search
  const filteredIndustry = industrySvcs.filter(s => {
    if (!search) return true;
    const q = search.toLowerCase();
    return s.title?.toLowerCase().includes(q)
      || s.description?.toLowerCase().includes(q)
      || s.industryUser?.user?.name?.toLowerCase().includes(q);
  });

  const filteredMkt = mktSvcs.filter(s => {
    if (!search) return true;
    const q = search.toLowerCase();
    return s.title?.toLowerCase().includes(q)
      || s.description?.toLowerCase().includes(q)
      || s.artist?.name?.toLowerCase().includes(q);
  });

  return (
    <>
      <Navbar />
      <main className="min-h-screen" style={{ background: 'var(--bg)', color: 'var(--text)' }}>

        {/* ── Hero Banner ─────────────────────────────────────────── */}
        <div className="relative overflow-hidden" style={{
          background: 'linear-gradient(135deg, rgba(201,162,39,0.07) 0%, rgba(160,232,124,0.05) 100%)',
          borderBottom: '1px solid var(--border)',
        }}>
          <div className="max-w-6xl mx-auto px-4 pt-20 pb-10">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
              <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold mb-4" style={{
                  background: 'rgba(201,162,39,0.12)', border: '1px solid rgba(201,162,39,0.25)', color: 'var(--gold)',
                }}>
                  <Zap size={11} /> HIRE & GET HIRED
                </div>
                <h1 className="text-3xl md:text-5xl font-black leading-tight mb-3">
                  Services Hub
                </h1>
                <p className="text-base md:text-lg max-w-xl" style={{ color: 'var(--text-muted)', lineHeight: 1.7 }}>
                  Industry professionals & artists offering their skills.
                  All payments through Vuka — <span style={{ color: 'var(--gold)' }}>secure, instant, transparent.</span>
                </p>
              </div>
              <div className="flex flex-wrap gap-3">
                {/* Fee transparency callout */}
                <div className="px-4 py-3 rounded-2xl text-sm" style={{
                  background: 'var(--surface)', border: '1px solid var(--border)',
                }}>
                  <p className="font-bold mb-0.5" style={{ color: 'var(--gold)' }}>Industry pros</p>
                  <p style={{ color: 'var(--text-muted)' }}>10% Vuka fee · You keep 90%</p>
                </div>
                <div className="px-4 py-3 rounded-2xl text-sm" style={{
                  background: 'var(--surface)', border: '1px solid var(--border)',
                }}>
                  <p className="font-bold mb-0.5" style={{ color: 'var(--green)' }}>Artist services</p>
                  <p style={{ color: 'var(--text-muted)' }}>5–15% fee by plan</p>
                </div>
              </div>
            </div>

            {/* Tab Bar */}
            <div className="flex gap-1 mt-8 p-1 rounded-2xl w-fit" style={{
              background: 'var(--surface)', border: '1px solid var(--border)',
            }}>
              {([
                { id: 'industry',    label: 'Industry Professionals', icon: Building2 },
                { id: 'marketplace', label: 'Artist Services',        icon: Mic2 },
              ] as const).map(t => (
                <button key={t.id} onClick={() => switchTab(t.id)}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold transition-all"
                  style={{
                    background: tab === t.id ? 'var(--gold)' : 'transparent',
                    color:      tab === t.id ? '#0a0a0a'   : 'var(--text-muted)',
                  }}>
                  <t.icon size={14} />
                  {t.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="max-w-6xl mx-auto px-4 py-8">

          {/* Search + Filter bar */}
          <div className="flex flex-col sm:flex-row gap-3 mb-6">
            <div className="relative flex-1">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} />
              <input className="input w-full pl-9 text-sm" placeholder="Search services, providers…"
                value={search} onChange={e => setSearch(e.target.value)} />
            </div>
            <select className="input text-sm" value={category} onChange={e => setCategory(e.target.value)}>
              {cats.map(c => <option key={c.value} value={c.value}>{c.label}</option>)}
            </select>
            {tab === 'industry' && (
              <select className="input text-sm" value={sort} onChange={e => setSort(e.target.value)}>
                <option value="price_asc">Price: Low → High</option>
                <option value="price_desc">Price: High → Low</option>
                <option value="newest">Newest</option>
              </select>
            )}
          </div>

          {/* ── Industry Tab ─────────────────────────────────────── */}
          {tab === 'industry' && (
            <>
              {userRole === 'industry' && (
                <div className="mb-6 p-4 rounded-2xl flex items-center justify-between gap-4" style={{
                  background: 'rgba(201,162,39,0.06)', border: '1px solid rgba(201,162,39,0.2)',
                }}>
                  <div>
                    <p className="font-bold text-sm" style={{ color: 'var(--gold)' }}>You're an industry professional</p>
                    <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                      Manage your listings and view incoming orders in your portal.
                    </p>
                  </div>
                  <Link href="/industry-dashboard"
                    className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold whitespace-nowrap"
                    style={{ background: 'var(--gold)', color: '#0a0a0a' }}>
                    My Portal <ChevronRight size={12} />
                  </Link>
                </div>
              )}

              {indLoading ? (
                <div className="flex justify-center py-20">
                  <Loader2 size={28} className="animate-spin" style={{ color: 'var(--gold)' }} />
                </div>
              ) : filteredIndustry.length === 0 ? (
                <EmptyState icon={<Building2 size={36} />} title="No industry services found"
                  sub="Try a different category or search term." />
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                  {filteredIndustry.map(svc => (
                    <IndustryCard
                      key={svc.id} svc={svc} userId={userId} userRole={userRole}
                      canAct={canAct}
                      showMsg={showMsg} msgMap={msgMap} inquiring={inquiring}
                      ordering={ordering} messaging={messaging}
                      doneMsgs={doneMsgs}
                      onShowMsg={id => setShowMsg(id)}
                      onMsgChange={(id, v) => setMsgMap(m => ({ ...m, [id]: v }))}
                      onInquire={sendInquiry}
                      onOrder={orderIndustryService}
                      onMessage={openMessage}
                    />
                  ))}
                </div>
              )}

              {/* CTA for industry professionals */}
              {!userRole || userRole === 'fan' ? (
                <div className="mt-12 p-8 rounded-3xl text-center" style={{
                  background: 'var(--surface)', border: '1px solid var(--border)',
                }}>
                  <Building2 size={32} className="mx-auto mb-4" style={{ color: 'var(--gold)' }} />
                  <h3 className="text-xl font-black mb-2">Are you an industry professional?</h3>
                  <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
                    List your services and get discovered by artists across Africa.
                    Vuka charges only 10% per order — you keep 90%.
                  </p>
                  <Link href="/auth/register?role=industry"
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-sm"
                    style={{ background: 'var(--gold)', color: '#0a0a0a' }}>
                    Create Industry Profile <ArrowRight size={14} />
                  </Link>
                </div>
              ) : null}
            </>
          )}

          {/* ── Marketplace Tab ───────────────────────────────────── */}
          {tab === 'marketplace' && (
            <>
              {mktLoading ? (
                <div className="flex justify-center py-20">
                  <Loader2 size={28} className="animate-spin" style={{ color: 'var(--green)' }} />
                </div>
              ) : filteredMkt.length === 0 ? (
                <EmptyState icon={<Mic2 size={36} />} title="No artist services found"
                  sub="Try a different category or search term." />
              ) : (
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
                  {filteredMkt.map(svc => (
                    <MarketplaceCard key={svc.id} service={svc}
                      onOrder={() => router.push(`/marketplace/${svc.id}`)} />
                  ))}
                </div>
              )}

              {/* Artist CTA */}
              {userRole === 'artist' || !userRole ? (
                <div className="mt-12 p-8 rounded-3xl text-center" style={{
                  background: 'var(--surface)', border: '1px solid var(--border)',
                }}>
                  <Mic2 size={32} className="mx-auto mb-4" style={{ color: 'var(--green)' }} />
                  <h3 className="text-xl font-black mb-2">Offer your skills</h3>
                  <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
                    List mixing, mastering, features, beats, and more.
                    Get paid securely through Vuka.
                  </p>
                  <Link href="/dashboard/services"
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-sm"
                    style={{ background: 'var(--green)', color: '#0a0a0a' }}>
                    Create a Service Listing <ArrowRight size={14} />
                  </Link>
                </div>
              ) : null}
            </>
          )}
        </div>
      </main>
    </>
  );
}

// ─── Industry Service Card ────────────────────────────────────────────────────
function IndustryCard({
  svc, userId, userRole, canAct,
  showMsg, msgMap, inquiring, ordering, messaging, doneMsgs,
  onShowMsg, onMsgChange, onInquire, onOrder, onMessage,
}: any) {
  const pm = PRICING_LABELS[svc.pricingModel] || '';
  const done = doneMsgs[svc.id];
  const isOpen = showMsg === svc.id;
  const isOwnService = svc.industryUser?.userId === userId || svc.industryUser?.user?.id === userId;

  // 10% fee transparency
  const price       = Number(svc.priceZAR);
  const platformFee = Math.round(price * 0.10 * 100) / 100;
  const providerNet = Math.round((price - platformFee) * 100) / 100;

  return (
    <div className="rounded-2xl flex flex-col overflow-hidden transition-all hover:translate-y-[-2px]"
      style={{ background: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 2px 12px rgba(0,0,0,0.15)' }}>

      {/* Gold top accent */}
      <div className="h-1" style={{ background: 'linear-gradient(90deg, var(--gold), rgba(201,162,39,0.3))' }} />

      <div className="p-6 flex flex-col flex-1">
        {/* Provider info */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center font-black text-sm flex-shrink-0"
            style={{ background: 'rgba(201,162,39,0.15)', color: 'var(--gold)' }}>
            {svc.industryUser?.user?.name?.[0]?.toUpperCase() || '?'}
          </div>
          <div className="min-w-0">
            <p className="font-bold text-sm truncate" style={{ color: 'var(--text)' }}>
              {svc.industryUser?.user?.name || 'Industry Professional'}
            </p>
            {svc.industryUser?.company || svc.industryUser?.companyName ? (
              <p className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>
                {svc.industryUser.company || svc.industryUser.companyName}
              </p>
            ) : null}
          </div>
          {svc.industryUser?.verified && (
            <div className="ml-auto flex-shrink-0">
              <span className="flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full"
                style={{ background: 'rgba(160,232,124,0.1)', color: 'var(--green)' }}>
                <CheckCircle size={9} /> Verified
              </span>
            </div>
          )}
        </div>

        {/* Category badge */}
        <div className="mb-3">
          <span className="text-[10px] font-bold px-2.5 py-1 rounded-full capitalize"
            style={{ background: 'rgba(201,162,39,0.1)', color: 'var(--gold)' }}>
            {svc.category || 'Service'}
          </span>
        </div>

        <h3 className="font-black mb-2 leading-snug" style={{ color: 'var(--text)' }}>{svc.title}</h3>
        {svc.description && (
          <p className="text-sm leading-relaxed flex-1 mb-4 line-clamp-3" style={{ color: 'var(--text-muted)' }}>
            {svc.description}
          </p>
        )}

        {/* Pricing row */}
        <div className="flex items-center gap-4 mb-1 flex-wrap">
          <span className="text-xl font-black" style={{ color: 'var(--gold)' }}>
            R{price.toLocaleString()} {pm}
          </span>
          <span className="flex items-center gap-1 text-xs" style={{ color: 'var(--text-muted)' }}>
            <Calendar size={11} /> {svc.deliveryDays}d delivery
          </span>
        </div>

        {/* Fee transparency — show to potential buyers */}
        {canAct && !isOwnService && (
          <div className="flex items-center gap-1.5 mb-4 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            <Shield size={10} style={{ color: 'var(--green)' }} />
            <span>Vuka fee: R{platformFee.toFixed(2)} · Provider gets: R{providerNet.toFixed(2)}</span>
          </div>
        )}

        {/* Actions */}
        {isOwnService ? (
          <Link href="/industry-dashboard" className="btn btn-secondary text-xs text-center">
            Manage listing
          </Link>
        ) : done === 'inquiry' ? (
          <div className="flex items-center gap-2 text-sm font-semibold" style={{ color: 'var(--green)' }}>
            <CheckCircle size={14} /> Inquiry sent
          </div>
        ) : done === 'ordered' ? (
          <div className="flex items-center gap-2 text-sm font-semibold" style={{ color: 'var(--green)' }}>
            <CheckCircle size={14} /> Payment initiated
          </div>
        ) : canAct ? (
          <div className="space-y-2">
            {isOpen ? (
              <>
                <textarea className="input w-full resize-none text-xs" rows={3}
                  placeholder="Tell them what you need (optional)…"
                  value={msgMap[svc.id] || ''}
                  onChange={e => onMsgChange(svc.id, e.target.value)} />
                <div className="flex gap-2">
                  <button onClick={() => onShowMsg(null)} className="btn btn-secondary flex-1 text-xs">Cancel</button>
                  {/* Inquiry (free, no payment) */}
                  <button onClick={() => onInquire(svc.id)} disabled={inquiring === svc.id}
                    className="btn btn-secondary flex-1 text-xs">
                    {inquiring === svc.id ? <Loader2 size={12} className="animate-spin" /> : <MessageSquare size={12} />}
                    Inquire
                  </button>
                  {/* Order (triggers Paystack payment) */}
                  <button onClick={() => onOrder(svc)} disabled={ordering === svc.id}
                    className="btn btn-primary flex-1 text-xs"
                    style={{ background: 'var(--gold)', color: '#0a0a0a' }}>
                    {ordering === svc.id ? <Loader2 size={12} className="animate-spin" /> : <ShoppingCart size={12} />}
                    Pay R{price.toLocaleString()}
                  </button>
                </div>
              </>
            ) : (
              <div className="flex gap-2">
                <button onClick={() => onShowMsg(svc.id)}
                  className="btn btn-primary flex-1 text-sm font-bold"
                  style={{ background: 'var(--gold)', color: '#0a0a0a' }}>
                  <ShoppingCart size={13} /> Order
                </button>
                <button onClick={() => onMessage(svc)} disabled={messaging === svc.id}
                  className="btn btn-secondary px-3 text-sm" title="Direct message">
                  {messaging === svc.id ? <Loader2 size={13} className="animate-spin" /> : <Send size={13} />}
                </button>
              </div>
            )}
          </div>
        ) : !userId ? (
          <Link href="/auth/login" className="btn btn-secondary text-sm text-center w-full">
            Sign in to hire
          </Link>
        ) : null}
      </div>
    </div>
  );
}

// ─── Marketplace Card ─────────────────────────────────────────────────────────
function MarketplaceCard({ service, onOrder }: { service: any; onOrder: () => void }) {
  const price = service.packages?.reduce((m: any, p: any) => (!m || p.price < m.price ? p : m), null)?.price
    ?? service.price ?? 0;

  return (
    <div className="rounded-2xl overflow-hidden flex flex-col cursor-pointer transition-all hover:translate-y-[-2px]"
      style={{ background: 'var(--surface)', border: '1px solid var(--border)', boxShadow: '0 2px 12px rgba(0,0,0,0.15)' }}
      onClick={onOrder}>
      <div className="h-1" style={{ background: 'linear-gradient(90deg, var(--green), rgba(160,232,124,0.3))' }} />
      <div className="p-5 flex flex-col flex-1 gap-3">
        <div className="flex items-center gap-2.5">
          {service.artist?.photoUrl
            ? <img src={service.artist.photoUrl} className="w-9 h-9 rounded-full object-cover flex-shrink-0" alt="" />
            : <div className="w-9 h-9 rounded-full flex-shrink-0 flex items-center justify-center font-black text-sm"
                style={{ background: 'rgba(160,232,124,0.15)', color: 'var(--green)' }}>
                {service.artist?.name?.[0] || 'A'}
              </div>
          }
          <div>
            <p className="text-xs font-bold" style={{ color: 'var(--green)' }}>{service.artist?.name}</p>
            {service.artist?.isVerified && (
              <span className="text-[10px] font-bold" style={{ color: 'var(--sky)' }}>✓ Verified</span>
            )}
          </div>
        </div>

        <div>
          <span className="text-[10px] font-bold px-2 py-0.5 rounded-full capitalize mb-2 inline-block"
            style={{ background: 'rgba(160,232,124,0.1)', color: 'var(--green)' }}>
            {service.category}
          </span>
          <h3 className="font-black text-sm leading-snug">{service.title}</h3>
          <p className="text-xs mt-1 line-clamp-2" style={{ color: 'var(--text-muted)' }}>{service.description}</p>
        </div>

        <div className="flex items-center gap-3 flex-wrap text-xs" style={{ color: 'var(--text-muted)' }}>
          {service.deliveryDays && (
            <span className="flex items-center gap-1"><Clock size={10} /> {service.deliveryDays}d</span>
          )}
          {service._count?.reviews > 0 && (
            <span className="flex items-center gap-1" style={{ color: '#e8c87c' }}>
              <Star size={10} fill="currentColor" /> {service._count.reviews}
            </span>
          )}
        </div>

        <div className="flex items-center justify-between mt-auto pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
          <div>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>From</p>
            <p className="text-xl font-black" style={{ color: 'var(--green)' }}>R{price.toFixed(0)}</p>
          </div>
          <button onClick={e => { e.stopPropagation(); onOrder(); }}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold"
            style={{ background: 'var(--green)', color: '#0a0a0a' }}>
            <ShoppingCart size={11} /> Order
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Empty State ──────────────────────────────────────────────────────────────
function EmptyState({ icon, title, sub }: { icon: React.ReactNode; title: string; sub: string }) {
  return (
    <div className="text-center py-20 rounded-2xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
      <div className="mx-auto mb-4 opacity-30">{icon}</div>
      <p className="font-bold mb-1" style={{ color: 'var(--text)' }}>{title}</p>
      <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{sub}</p>
    </div>
  );
}
