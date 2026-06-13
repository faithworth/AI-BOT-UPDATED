import Link from 'next/link';
import { Music2 } from 'lucide-react';

export const metadata = { title: 'Privacy Policy — Vuka', description: 'How Vuka collects and uses your data.' };

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <section className="mb-8">
    <h2 className="text-lg font-bold mb-3" style={{ color: 'var(--text)', fontFamily: 'var(--font-display)' }}>{title}</h2>
    <div className="space-y-3 text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{children}</div>
  </section>
);

export default function PrivacyPage() {
  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <header className="px-6 py-4 flex items-center gap-3" style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
        <Link href="/" className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: 'var(--sky)' }}><Music2 size={13} className="text-white" /></div>
          <span className="font-bold" style={{ color: 'var(--text)' }}>Vuka</span>
        </Link>
        <span style={{ color: 'var(--border)' }}>/</span>
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Privacy Policy</span>
      </header>
      <main className="max-w-3xl mx-auto px-6 py-12">
        <h1 className="text-3xl font-black mb-2" style={{ color: 'var(--text)', fontFamily: 'var(--font-display)' }}>Privacy Policy</h1>
        <p className="text-sm mb-10" style={{ color: 'var(--text-muted)' }}>Last updated: January 2025 · POPIA & GDPR aware</p>

        <Section title="1. Data We Collect">
          <p><strong style={{ color: 'var(--text)' }}>Account data:</strong> name, email address, and role when you register.</p>
          <p><strong style={{ color: 'var(--text)' }}>Purchase data:</strong> transaction amounts, item purchased, licence type, and download activity.</p>
          <p><strong style={{ color: 'var(--text)' }}>Content data:</strong> files you upload (beats, releases, artwork) stored in Cloudflare R2.</p>
          <p><strong style={{ color: 'var(--text)' }}>Usage data:</strong> pages visited, plays, and interactions — collected anonymously for platform improvements.</p>
          <p>We do not collect payment card details. All payment processing is handled by Paystack.</p>
        </Section>

        <Section title="2. How We Use Your Data">
          <p>We use your data to: operate your account; process and fulfil purchases; send transactional emails (receipts, download links); notify you of important account changes; and improve the platform.</p>
          <p>We do not sell your personal data to any third party. We do not use your data for targeted advertising.</p>
        </Section>

        <Section title="3. Data Sharing">
          <p>Your data is shared only with: (a) Supabase — authentication and session management; (b) Cloudflare R2 — file storage; (c) Paystack — payment processing; (d) Resend — transactional email delivery. All providers are contractually bound to process data only as instructed.</p>
          <p>Artist names and public profile information are visible to other users as part of the marketplace experience.</p>
        </Section>

        <Section title="4. Data Retention">
          <p>Account data is retained for as long as your account is active. Purchase records are retained for 7 years for accounting and legal compliance. You may request deletion of your account and associated data at any time by emailing us.</p>
        </Section>

        <Section title="5. Your Rights (POPIA & GDPR)">
          <p>You have the right to: access the personal data we hold about you; correct inaccurate data; request deletion (subject to legal retention requirements); object to processing; and data portability. Submit requests to support@vuka.co.za.</p>
        </Section>

        <Section title="6. Cookies">
          <p>We use only essential session cookies required for authentication. We do not use tracking or advertising cookies.</p>
        </Section>

        <Section title="7. Security">
          <p>All data in transit is encrypted via TLS. Files are stored in private Cloudflare R2 buckets. Download links expire and are single-use. We follow industry-standard security practices.</p>
        </Section>

        <Section title="8. Contact">
          <p>For any privacy concerns, contact us at <a href="mailto:support@vuka.co.za" className="underline" style={{ color: 'var(--sky)' }}>support@vuka.co.za</a>.</p>
        </Section>

        <div className="mt-10 pt-8" style={{ borderTop: '1px solid var(--border)' }}>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            <Link href="/legal/terms" className="underline" style={{ color: 'var(--sky)' }}>Terms of Service</Link>
            {' · '}<Link href="/legal/refunds" className="underline" style={{ color: 'var(--sky)' }}>Refund Policy</Link>
            {' · '}<Link href="/legal/dmca" className="underline" style={{ color: 'var(--sky)' }}>DMCA</Link>
          </p>
        </div>
      </main>
    </div>
  );
}
