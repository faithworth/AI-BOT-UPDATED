// ============================================================
// VUKA CMS Core Library
// ============================================================
import prisma from '@/lib/prisma';

export const CMS_ADMIN_ROLES  = ['owner', 'super_admin', 'admin'];
export const CMS_EDITOR_ROLES = [...CMS_ADMIN_ROLES, 'moderator'];

export const canAccessCms = (role: string) => CMS_EDITOR_ROLES.includes(role);
export const canPublish   = (role: string) => CMS_ADMIN_ROLES.includes(role);
export const canDelete    = (role: string) => ['owner', 'super_admin'].includes(role);

// ── Block catalogue ──────────────────────────────────────────
export const BLOCK_TYPES = [
  { type: 'hero',          label: 'Hero Banner',       icon: '🎯', desc: 'Full-width headline with CTA buttons' },
  { type: 'text',          label: 'Text',              icon: '📝', desc: 'Paragraph or heading text' },
  { type: 'rich_text',     label: 'Rich Text',         icon: '✍️',  desc: 'Formatted HTML content' },
  { type: 'image',         label: 'Image',             icon: '🖼️',  desc: 'Single image with alt & caption' },
  { type: 'video',         label: 'Video Embed',       icon: '🎥', desc: 'YouTube or Vimeo embed' },
  { type: 'cta',           label: 'Call to Action',    icon: '🚀', desc: 'Button row or action section' },
  { type: 'features_grid', label: 'Features Grid',     icon: '⚡', desc: 'Grid of feature cards with icons' },
  { type: 'pricing',       label: 'Pricing Cards',     icon: '💰', desc: 'Pricing tier cards' },
  { type: 'artists_grid',  label: 'Featured Artists',  icon: '🎵', desc: 'Showcases featured artists' },
  { type: 'stats',         label: 'Stats Row',         icon: '📊', desc: 'Highlighted metrics row' },
  { type: 'faq',           label: 'FAQ',               icon: '❓', desc: 'Collapsible Q&A' },
  { type: 'banner',        label: 'Banner / Alert',    icon: '📢', desc: 'Coloured announcement strip' },
  { type: 'testimonials',  label: 'Testimonials',      icon: '💬', desc: 'Quote cards' },
  { type: 'spacer',        label: 'Spacer',            icon: '↕️',  desc: 'Vertical spacing' },
  { type: 'html',          label: 'Custom HTML',       icon: '🔧', desc: 'Raw HTML (admins only)' },
] as const;

export type BlockType = typeof BLOCK_TYPES[number]['type'];

// ── Default content per block type ───────────────────────────
export function defaultBlockContent(type: BlockType): Record<string, unknown> {
  switch (type) {
    case 'hero':
      return {
        badge: "Africa's independent music platform",
        headline: "Your music.\nYour terms.\nYour money.",
        subheadline: 'Sell beats and releases directly to your fans — in South Africa and worldwide. Keep up to 95% of every sale.',
        cta_primary:   { label: "Start Selling — It's Free", href: '/auth/register' },
        cta_secondary: { label: 'Browse the Store',          href: '/store' },
        stats: [
          { value: '85%', label: 'Artist keeps (Free)' },
          { value: 'ZAR', label: 'Paid in Rands'       },
          { value: '95%', label: 'Artist keeps (Label)' },
        ],
      };
    case 'text':
      return { heading: '', body: 'Enter your text here.', align: 'left' };
    case 'rich_text':
      return { html: '<p>Enter your content here.</p>' };
    case 'image':
      return { src: '', alt: '', caption: '', rounded: true };
    case 'video':
      return { url: '', caption: '' };
    case 'cta':
      return {
        heading: 'Ready to get started?',
        subheading: '',
        buttons: [
          { label: 'Get Started Free', href: '/auth/register', variant: 'primary' },
          { label: 'Browse Store',     href: '/store',         variant: 'secondary' },
        ],
      };
    case 'features_grid':
      return {
        heading: 'Everything you need',
        subheading: '',
        columns: 4,
        features: [
          { icon: '⚡', title: 'Instant Payouts',       desc: 'Get paid to your SA bank account within 48 hours.' },
          { icon: '🎯', title: 'Beat Licensing',        desc: 'Sell non-exclusive, exclusive and custom licences.' },
          { icon: '💎', title: 'Keep More',             desc: 'Up to 95% per sale — more than any competitor.' },
          { icon: '🌍', title: 'Global Reach',          desc: 'Reach fans in SA and worldwide with one upload.' },
        ],
      };
    case 'pricing':
      return {
        heading: 'Honest pricing. Always.',
        subheading: "Start free. Upgrade when you're ready to keep more.",
        tiers: [
          {
            name: 'Free', price: 'R0', period: 'forever', keep: '85%', highlight: false,
            features: ['Up to 2 releases/month', 'Beat store & licensing', 'Paystack checkout'],
            cta: { label: 'Get Started Free', href: '/auth/register' },
          },
          {
            name: 'Pro', price: 'R249', period: 'per month', keep: '92%', highlight: true,
            features: ['Unlimited releases', '8% platform fee', 'Advanced analytics'],
            cta: { label: 'Start Pro', href: '/auth/register' },
          },
          {
            name: 'Label', price: 'R999', period: 'per month', keep: '95%', highlight: false,
            features: ['Unlimited releases', '5% platform fee', 'Multi-artist roster'],
            cta: { label: 'Start Label', href: '/auth/register' },
          },
        ],
      };
    case 'artists_grid':
      return { heading: 'Featured Artists', subheading: 'Discover incredible talent on Vuka.', max: 6 };
    case 'stats':
      return {
        items: [
          { value: '10,000+', label: 'Artists on Vuka'   },
          { value: '50,000+', label: 'Tracks uploaded'    },
          { value: 'R2M+',    label: 'Paid to artists'   },
        ],
      };
    case 'faq':
      return {
        heading: 'Frequently Asked Questions',
        items: [
          { q: 'How do I get paid?',          a: 'Payments are processed via Paystack. Money is sent to your SA bank account within 48 hours of a sale.' },
          { q: 'What is the platform fee?',   a: 'Free plan: 15%. Pro: 8%. Label: 5%. Fees are deducted automatically at checkout.' },
          { q: 'Can I sell internationally?', a: 'Yes — Vuka supports international buyers via card. Your earnings are converted to ZAR.' },
        ],
      };
    case 'banner':
      return { text: 'New announcement coming soon!', variant: 'info', dismissible: true, link: '', linkLabel: '' };
    case 'testimonials':
      return {
        heading: 'What artists say',
        items: [
          { quote: 'Vuka changed how I sell my music. I keep so much more.', author: 'DJ Khali',   role: 'Hip-Hop Producer, Cape Town' },
          { quote: "Finally a platform built for African artists.",           author: 'Amara Soul', role: 'Afrobeats Artist, Durban' },
        ],
      };
    case 'spacer':
      return { height: 64 };
    case 'html':
      return { code: '<!-- Custom HTML -->' };
    default:
      return {};
  }
}

// ── DB helpers ───────────────────────────────────────────────
export async function getCmsPage(idOrSlug: string) {
  // Try by ID first, then slug
  const page = await prisma.cmsPage.findFirst({
    where: { OR: [{ id: idOrSlug }, { slug: idOrSlug }] },
    include: {
      blocks: { orderBy: { order: 'asc' } },
      collaborators: {
        include: { user: { select: { id: true, name: true, email: true, role: true } } },
      },
      comments: { orderBy: { createdAt: 'desc' } },
    },
  });
  return page;
}

export async function getPublishedPage(slug: string) {
  return prisma.cmsPage.findFirst({
    where: { slug, status: 'published' },
    include: { blocks: { where: { isVisible: true }, orderBy: { order: 'asc' } } },
  });
}

export async function getAllPages() {
  return prisma.cmsPage.findMany({
    orderBy: [{ isSystem: 'desc' }, { updatedAt: 'desc' }],
    select: {
      id: true, slug: true, title: true, description: true,
      status: true, isSystem: true, publishedAt: true, updatedAt: true, createdById: true,
      _count: { select: { blocks: true, collaborators: true, comments: true } },
    },
  });
}

export async function getFeaturedArtists() {
  return prisma.featuredArtist.findMany({
    where: { isVisible: true },
    orderBy: { order: 'asc' },
    include: {
      artist: {
        select: {
          id: true, slug: true, name: true, photoUrl: true, coverUrl: true,
          genreTags: true, city: true, country: true, isVerified: true, totalPlays: true,
          _count: { select: { beats: true, releases: true, followers: true } },
        },
      },
    },
  });
}

export async function getAllFeaturedArtists() {
  return prisma.featuredArtist.findMany({
    orderBy: { order: 'asc' },
    include: {
      artist: {
        select: {
          id: true, slug: true, name: true, photoUrl: true,
          genreTags: true, city: true, isVerified: true,
          _count: { select: { followers: true, beats: true } },
        },
      },
    },
  });
}

export async function createRevision(pageId: string, createdById: string, summary: string) {
  const blocks = await prisma.cmsBlock.findMany({
    where: { pageId },
    orderBy: { order: 'asc' },
  });
  return prisma.cmsRevision.create({
    data: {
      pageId,
      createdById,
      summary,
      blocks: blocks as unknown as import('@prisma/client').Prisma.JsonArray,
    },
  });
}

export async function getRevisions(pageId: string) {
  return prisma.cmsRevision.findMany({
    where: { pageId },
    orderBy: { createdAt: 'desc' },
    take: 30,
    select: { id: true, summary: true, createdAt: true, createdById: true },
  });
}
