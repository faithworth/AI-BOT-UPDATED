import Link from 'next/link';
import { Music2 } from 'lucide-react';

export const metadata = { title: 'Terms of Service — Vuka', description: 'Vuka platform terms of service.' };

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <section className="mb-8">
    <h2 className="text-lg font-bold mb-3" style={{ color: 'var(--text)', fontFamily: 'var(--font-display)' }}>{title}</h2>
    <div className="space-y-3 text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{children}</div>
  </section>
);

export default function TermsPage() {
  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>
      <header className="px-6 py-4 flex items-center gap-3" style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
        <Link href="/" className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: 'var(--sky)' }}><Music2 size={13} className="text-white" /></div>
          <span className="font-bold" style={{ color: 'var(--text)' }}>Vuka</span>
        </Link>
        <span style={{ color: 'var(--border)' }}>/</span>
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Terms of Service</span>
      </header>
      <main className="max-w-3xl mx-auto px-6 py-12">
        <h1 className="text-3xl font-black mb-2" style={{ color: 'var(--text)', fontFamily: 'var(--font-display)' }}>Terms of Service</h1>
        <p className="text-sm mb-10" style={{ color: 'var(--text-muted)' }}>Last updated: January 2025 · Governing law: Republic of South Africa</p>

        <Section title="1. About Vuka">
          <p>Vuka ("we", "us", "the platform") is a digital music marketplace operated by Faithworth, Republic of South Africa. These Terms govern your use of vuka.co.za and all related services.</p>
          <p>By creating an account or making a purchase, you confirm you are at least 18 years old and agree to these Terms in full.</p>
        </Section>

        <Section title="2. User Accounts">
          <p>You are responsible for keeping your credentials secure. You may not share, sell, or transfer your account. We reserve the right to suspend or terminate any account found to be in violation of these Terms.</p>
          <p>Artists must provide accurate information about themselves and the content they upload. Impersonation of another artist or person is prohibited.</p>
        </Section>

        <Section title="3. Content & Intellectual Property">
          <p>Artists retain full copyright ownership of all music, beats, and media uploaded to Vuka. By uploading, you grant Vuka a non-exclusive, royalty-free licence to display, stream, and promote your content on the platform.</p>
          <p>You warrant that you own or have the right to sell all content you upload. Content that infringes third-party copyright, contains unlicensed samples, or violates any applicable law is prohibited and will be removed.</p>
          <p>Buyers receive a licence to use purchased content in accordance with the licence tier selected at checkout. Licences are non-transferable and do not grant ownership of the underlying copyright.</p>
        </Section>

        <Section title="4. Exclusive Licences">
          <p>When a buyer purchases an Exclusive Licence, that beat or content is permanently removed from sale. The artist may not re-list, re-sell, or grant any further licences for that specific work to any other party. This is a binding commitment enforced at the platform level.</p>
          <p>Exclusive licence purchases are final. No refunds are issued once the exclusive lock has been applied.</p>
        </Section>

        <Section title="5. Payments & Fees">
          <p>All prices are displayed in South African Rand (ZAR). Payments are processed via Paystack for domestic transactions. International payment options may be added in future.</p>
          <p>Vuka charges a small platform fee on each transaction to cover operational costs. This fee is deducted before funds are disbursed to the artist. Artists can view net earnings in their Payouts dashboard.</p>
          <p>Vuka does not store payment card information. All payment data is handled by Paystack in accordance with PCI DSS standards.</p>
        </Section>

        <Section title="6. No Refund Policy">
          <p>All sales of digital goods on Vuka are final. We do not offer refunds once a purchase is confirmed and download access has been granted. This policy is consistent with the Electronic Communications and Transactions Act 25 of 2002 (ECT Act), which permits sellers to exclude the right of return for digital goods delivered immediately upon purchase.</p>
          <p>If you believe you were charged incorrectly or did not receive access to your purchase, contact us at support@vuka.co.za within 7 days and we will investigate.</p>
        </Section>

        <Section title="7. Prohibited Conduct">
          <p>You may not: (a) upload content you do not own; (b) use automated tools to scrape, download, or rip content; (c) attempt to circumvent download protection or DRM measures; (d) resell or redistribute purchased digital files; (e) upload malware, spam, or illegal content; (f) harass, threaten, or abuse other users.</p>
        </Section>

        <Section title="8. DMCA & Takedowns">
          <p>Vuka respects intellectual property rights. If you believe your work has been uploaded without authorisation, submit a DMCA notice at <Link href="/legal/dmca" className="underline" style={{ color: 'var(--sky)' }}>/legal/dmca</Link>. We will investigate and act within 72 hours of receiving a valid notice.</p>
        </Section>

        <Section title="9. Limitation of Liability">
          <p>To the maximum extent permitted by South African law, Vuka's liability for any claim arising from use of the platform is limited to the amount you paid in the 30 days preceding the claim. We are not liable for loss of profits, data loss, or consequential damages.</p>
        </Section>

        <Section title="10. Governing Law">
          <p>These Terms are governed by the laws of the Republic of South Africa. Any disputes shall be submitted to the jurisdiction of the South African courts.</p>
        </Section>

        <Section title="11. Changes to Terms">
          <p>We may update these Terms at any time. Continued use of the platform after changes are posted constitutes acceptance. We will notify registered users of material changes by email.</p>
        </Section>

        <div className="mt-10 pt-8" style={{ borderTop: '1px solid var(--border)' }}>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Questions? Email <a href="mailto:support@vuka.co.za" className="underline" style={{ color: 'var(--sky)' }}>support@vuka.co.za</a>
            {' · '}<Link href="/legal/privacy" className="underline" style={{ color: 'var(--sky)' }}>Privacy Policy</Link>
            {' · '}<Link href="/legal/refunds" className="underline" style={{ color: 'var(--sky)' }}>Refund Policy</Link>
            {' · '}<Link href="/legal/dmca" className="underline" style={{ color: 'var(--sky)' }}>DMCA</Link>
          </p>
        </div>
      </main>
    </div>
  );
}
