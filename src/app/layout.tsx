import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Vuka — Release. Distribute. Earn. Own.',
  description:
    "Vuka is Africa's independent music distribution platform. Upload your music, distribute to Spotify, Apple Music, Boomplay, Audiomack and 30+ platforms. Keep up to 100% of your royalties. Built for South African and African independent artists.",
  keywords: [
    'vuka', 'vuka distro', 'music distribution south africa',
    'south african music platform', 'african music distribution',
    'distribute music africa', 'independent music artist sa',
    'amapiano distribution', 'gqom beats', 'afrobeats distribution',
    'paystack music', 'boomplay distribution', 'audiomack distribution',
  ],
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'https://www.vuka.co.za'),
  alternates: { canonical: process.env.NEXT_PUBLIC_APP_URL || 'https://www.vuka.co.za' },
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true, 'max-image-preview': 'large' },
  },
  openGraph: {
    title: 'Vuka — Release. Distribute. Earn. Own.',
    description:
      "Africa's independent music distribution platform. Upload once, distribute everywhere. Keep up to 100% royalties. Paystack payments.",
    url: process.env.NEXT_PUBLIC_APP_URL || 'https://www.vuka.co.za',
    siteName: 'Vuka',
    type: 'website',
    locale: 'en_ZA',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Vuka — Release. Distribute. Earn. Own.',
    description:
      "Africa's independent music distribution platform. Upload once, distribute everywhere. Keep up to 100% royalties.",
  },
  icons: { icon: '/favicon.svg' },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
        {/* Vuka Design System fonts — Syne (headings), DM Sans (body), JetBrains Mono (numbers/code) */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=JetBrains+Mono:wght@400;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
