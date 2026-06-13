'use client';
import Link from 'next/link';
import Navbar from '@/components/Navbar';
import {
  ArrowRight, CheckCircle, Megaphone, Building2, Briefcase,
  Radio, Search, Music, Camera, Headphones, Scale, Star,
  DollarSign, Users, Clock, Shield, Zap, TrendingUp,
} from 'lucide-react';

const SERVICE_TYPES = [
  { icon: Megaphone,  label: 'Promotion & Marketing',   desc: 'Social campaigns, playlist pitching, press releases, radio plugging.' },
  { icon: Building2,  label: 'Distribution & Publishing', desc: 'Get artists onto streaming platforms, collect royalties, handle publishing admin.' },
  { icon: Radio,      label: 'Sync & Licensing',         desc: 'Place music in film, TV, adverts, and video games. Handle sync clearance.' },
  { icon: Briefcase,  label: 'Artist Management',        desc: 'Day-to-day artist management, bookings, strategy, and career development.' },
  { icon: Search,     label: 'Talent Scouting',          desc: 'Scout and refer artists to labels, festivals, and brand partnerships.' },
  { icon: Star,       label: 'Sponsorship & Brand Deals', desc: 'Connect artists with brands for endorsements, collaborations, and campaigns.' },
  { icon: Scale,      label: 'Legal & Contracts',        desc: 'Draft and review music contracts, licensing agreements, and IP protection.' },
  { icon: Camera,     label: 'Photography & Videography', desc: 'Press photos, EPKs, music video production, and content creation.' },
  { icon: Headphones, label: 'Mixing & Mastering',       desc: 'Professional audio finishing — make every release sound release-ready.' },
];

const HOW_IT_WORKS = [
  {
    n: '01', t: 'Create your profile',
    d: 'Sign up as an Industry Professional. Add your company, role, and a short bio about what you offer.',
  },
  {
    n: '02', t: 'List your services',
    d: 'Create service listings with your price, turnaround time, and exactly what artists get when they hire you.',
  },
  {
    n: '03', t: 'Artists find and hire you',
    d: 'Artists on Vuka browse the Services hub. They pay directly through Vuka — payment is immediate, no chasing invoices.',
  },
  {
    n: '04', t: 'You receive 90% instantly',
    d: 'Vuka deducts 10% as a platform fee. The remaining 90% is queued for your next payout. No admin, no delays.',
  },
];

const WHY = [
  { icon: DollarSign, t: 'You keep 90% of every order',      d: 'Vuka charges a flat 10% platform fee on industry service orders. You set your own price — Vuka never dictates rates.' },
  { icon: Zap,        t: 'Instant payment confirmation',      d: 'Artists pay via Paystack at checkout. Payment is confirmed immediately — no waiting, no invoicing back and forth.' },
  { icon: Users,      t: 'Direct access to active artists',  d: 'All artists on Vuka are already selling music. They have revenue and they invest in their careers.' },
  { icon: Shield,     t: 'Vuka holds payment safely',        d: 'Payments go through Vuka\'s escrow — artists can\'t cancel a confirmed payment, and you always get paid.' },
  { icon: Music,      t: 'Africa-first community',           d: 'Built for Amapiano, Afrobeats, Gqom, Hip Hop, Kwaito and more. Work with artists who need you.' },
  { icon: TrendingUp, t: 'Browse and find artists too',      d: 'Industry professionals can browse all artists on Vuka, filter by genre, and send deals or offers directly.' },
];

export default function IndustryPage() {
  return (
    <>
      <Navbar />
      <main style={{ background: 'var(--bg)', color: 'var(--text)' }}>

        {/* HERO */}
        <section className="relative min-h-[88vh] flex items-center justify-center overflow-hidden pt-20 pb-24 px-4">
          <div className="absolute inset-0 pointer-events-none" style={{
            background: 'radial-gradient(ellipse 70% 55% at 50% -10%, rgba(201,162,39,0.10) 0%, transparent 70%)',
          }} />
          <div className="relative z-10 max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold mb-8 tracking-wide" style={{
              background: 'rgba(201,162,39,0.1)',
              border: '1px solid rgba(201,162,39,0.25)',
              color: 'var(--gold)',
            }}>
              <Briefcase size={12} /> FOR INDUSTRY PROFESSIONALS
            </div>

            <h1 className="text-4xl md:text-6xl font-black mb-6 leading-tight" style={{ color: 'var(--text)' }}>
              List your services.<br />
              <span style={{ color: 'var(--gold)' }}>Get paid automatically.</span>
            </h1>

            <p className="text-lg md:text-xl mb-4 max-w-2xl mx-auto" style={{ color: 'var(--text-muted)', lineHeight: 1.7 }}>
              Vuka is where Africa's artists come to sell their music. Create a profile, list your professional services,
              and get hired directly by artists — with payment handled instantly through the platform.
            </p>

            {/* Fee callout */}
            <div className="inline-flex items-center gap-3 px-5 py-3 rounded-2xl mb-8" style={{
              background: 'rgba(201,162,39,0.08)', border: '1px solid rgba(201,162,39,0.2)',
            }}>
              <DollarSign size={16} style={{ color: 'var(--gold)' }} />
              <span className="text-sm font-bold" style={{ color: 'var(--text)' }}>
                Vuka charges <span style={{ color: 'var(--gold)' }}>10% per order</span> · You keep 90% · No subscription fee
              </span>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/auth/register?role=industry"
                className="btn btn-primary text-base px-8 py-4 font-bold inline-flex items-center gap-2"
                style={{ background: 'var(--gold)', color: '#0a0a0a' }}>
                Create Industry Profile <ArrowRight size={18} />
              </Link>
              <Link href="/services"
                className="btn btn-secondary text-base px-8 py-4 font-bold inline-flex items-center gap-2">
                Browse Services
              </Link>
            </div>
          </div>
        </section>

        {/* WHAT YOU CAN OFFER */}
        <section className="py-20 px-4" style={{ background: 'var(--surface)' }}>
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-14">
              <h2 className="text-3xl md:text-4xl font-black mb-4" style={{ color: 'var(--text)' }}>
                What services can you list?
              </h2>
              <p className="text-lg" style={{ color: 'var(--text-muted)' }}>
                Anything that helps artists grow, sound better, or reach more people.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              {SERVICE_TYPES.map(s => (
                <div key={s.label} className="p-5 rounded-2xl transition-all hover:border-yellow-500/30"
                  style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-3"
                    style={{ background: 'rgba(201,162,39,0.1)' }}>
                    <s.icon size={18} style={{ color: 'var(--gold)' }} />
                  </div>
                  <p className="font-bold mb-1.5" style={{ color: 'var(--text)' }}>{s.label}</p>
                  <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{s.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* HOW IT WORKS */}
        <section className="py-20 px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-black text-center mb-14" style={{ color: 'var(--text)' }}>
              How it works
            </h2>
            <div className="space-y-6">
              {HOW_IT_WORKS.map(h => (
                <div key={h.n} className="flex gap-6 p-6 rounded-2xl"
                  style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                  <div className="text-3xl font-black flex-shrink-0 w-12 text-center leading-none"
                    style={{ color: 'var(--gold)', opacity: 0.5 }}>{h.n}</div>
                  <div>
                    <p className="font-bold mb-1" style={{ color: 'var(--text)' }}>{h.t}</p>
                    <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{h.d}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* FEE TRANSPARENCY */}
        <section className="py-16 px-4" style={{ background: 'rgba(201,162,39,0.04)', borderTop: '1px solid rgba(201,162,39,0.1)', borderBottom: '1px solid rgba(201,162,39,0.1)' }}>
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-2xl md:text-3xl font-black mb-6" style={{ color: 'var(--text)' }}>
              Simple, transparent pricing
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
              {[
                { label: 'Order amount', example: 'R5,000', color: 'var(--text)' },
                { label: 'Vuka fee (10%)', example: '− R500', color: 'var(--gold)' },
                { label: 'You receive', example: 'R4,500', color: 'var(--green)' },
              ].map(item => (
                <div key={item.label} className="p-5 rounded-2xl" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                  <p className="text-sm mb-2" style={{ color: 'var(--text-muted)' }}>{item.label}</p>
                  <p className="text-2xl font-black" style={{ color: item.color }}>{item.example}</p>
                </div>
              ))}
            </div>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              The fee is calculated automatically — no invoices to send, no chasing payments.
              Artists pay at checkout; Vuka routes the net amount to your payout queue.
            </p>
          </div>
        </section>

        {/* WHY VUKA */}
        <section className="py-20 px-4" style={{ background: 'var(--surface)' }}>
          <div className="max-w-5xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-black text-center mb-14" style={{ color: 'var(--text)' }}>
              Why list on Vuka?
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {WHY.map(w => (
                <div key={w.t} className="flex gap-4 p-6 rounded-2xl"
                  style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{ background: 'rgba(201,162,39,0.1)' }}>
                    <w.icon size={18} style={{ color: 'var(--gold)' }} />
                  </div>
                  <div>
                    <p className="font-bold mb-1" style={{ color: 'var(--text)' }}>{w.t}</p>
                    <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{w.d}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="py-24 px-4 text-center">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-black mb-4" style={{ color: 'var(--text)' }}>
              Ready to work with Africa's artists?
            </h2>
            <p className="text-lg mb-10" style={{ color: 'var(--text-muted)' }}>
              Create your free profile and start listing your services today.
              Payments happen automatically — you focus on the work.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
              <Link href="/auth/register?role=industry"
                className="btn btn-primary text-base px-10 py-4 font-bold inline-flex items-center gap-2"
                style={{ background: 'var(--gold)', color: '#0a0a0a' }}>
                Get Started — It's Free <ArrowRight size={18} />
              </Link>
              <Link href="/browse-artists"
                className="btn btn-secondary text-base px-8 py-4 font-bold inline-flex items-center gap-2">
                Browse Artists <Users size={16} />
              </Link>
            </div>
            <div className="flex items-center justify-center gap-6 text-sm flex-wrap" style={{ color: 'var(--text-muted)' }}>
              {['No subscription fee', 'You set your price', '10% platform fee only'].map(t => (
                <span key={t} className="flex items-center gap-1.5">
                  <CheckCircle size={14} style={{ color: 'var(--green)' }} /> {t}
                </span>
              ))}
            </div>
          </div>
        </section>

      </main>
    </>
  );
}
