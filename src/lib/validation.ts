/**
 * VUKA — Shared Zod Validation Schemas (Phase 8)
 *
 * Single source of truth for all input validation schemas.
 * Import into route handlers — never trust raw req.json() without parsing.
 *
 * Usage:
 *   import { schemas } from '@/lib/validation';
 *   const parsed = schemas.beat.upload.safeParse(await req.json());
 *   if (!parsed.success) return validationError(parsed.error);
 */

import { z } from 'zod';

// ── Helpers ────────────────────────────────────────────────────────────────

/** Sanitise a plain text field — trim whitespace, max length */
const safeText = (max: number) =>
  z.string()
    .max(max)
    .trim();

/** CUID-shaped ID (26 chars, starts with c) */
const cuid = () =>
  z.string().regex(/^c[a-z0-9]{24,}$/, 'Invalid ID format');

/** ISO-8601 datetime string */
const isoDate = () =>
  z.string().datetime({ message: 'Must be a valid ISO-8601 datetime' });

/** URL — must start with https */
const httpsUrl = () =>
  z.string()
    .url()
    .refine((u) => u.startsWith('https://'), 'URL must use HTTPS');

/** Positive decimal amount (up to 2 decimal places) */
const money = () =>
  z.number()
    .positive('Amount must be positive')
    .multipleOf(0.01, 'Maximum 2 decimal places');

// ── Auth ───────────────────────────────────────────────────────────────────

const authRegister = z.object({
  email:       z.string().email().max(254).trim().toLowerCase(),
  name:        safeText(100).min(2, 'Name too short'),
  password:    z.string().min(10, 'Password must be at least 10 characters').max(128),
  role:        z.enum(['artist', 'fan', 'industry']).default('artist'),
  inviteCode:  z.string().max(64).optional(),
});

const authLogin = z.object({
  email:    z.string().email().max(254).trim().toLowerCase(),
  password: z.string().min(1).max(128),
});

const magicLinkRequest = z.object({
  email: z.string().email().max(254).trim().toLowerCase(),
});

// ── Artist profile ─────────────────────────────────────────────────────────

const artistUpdate = z.object({
  name:        safeText(100).min(2).optional(),
  bio:         safeText(1000).optional(),
  city:        safeText(100).optional(),
  country:     z.string().length(2).toUpperCase().optional(),
  genreTags:   z.array(safeText(50)).max(10).optional(),
  photoUrl:    httpsUrl().optional().nullable(),
  coverUrl:    httpsUrl().optional().nullable(),
  currency:    z.enum(['ZAR', 'USD', 'EUR', 'GBP', 'NGN', 'KES', 'GHS']).optional(),
  socialLinks: z.record(z.string().max(20), httpsUrl()).optional(),
  paystackRecipient: z.string().max(20).trim().optional().nullable(),
});

// ── Beat upload ────────────────────────────────────────────────────────────

const VALID_GENRES = [
  'Amapiano', 'Gqom', 'Afrobeats', 'Hip-Hop', 'R&B', 'Pop',
  'Drill', 'Gospel', 'Jazz', 'Electronic', 'House', 'Kwaito',
  'Neo-Soul', 'Afro-Soul', 'Other',
] as const;

const beatUpload = z.object({
  title:         safeText(200).min(1, 'Title required'),
  bpm:           z.number().int().min(40).max(300).optional(),
  key:           safeText(10).optional(),
  genre:         z.enum(VALID_GENRES).optional(),
  tags:          z.array(safeText(50)).max(20).default([]),
  isExplicit:    z.boolean().default(false),
  isExclusive:   z.boolean().default(false),
  audioFileKey:  z.string().max(500).min(1, 'Audio file key required'),
  imageFileKey:  z.string().max(500).optional(),
  price:         money().min(0, 'Price cannot be negative').optional(),
  licenseType:   z.enum(['basic', 'premium', 'exclusive', 'free']).default('basic'),
});

const beatUpdate = beatUpload.partial().omit({ audioFileKey: true });

// ── Release ────────────────────────────────────────────────────────────────

const releaseCreate = z.object({
  title:          safeText(200).min(1, 'Title required'),
  type:           z.enum(['single', 'ep', 'album', 'mixtape']),
  genres:         z.array(z.enum(VALID_GENRES)).min(1).max(3),
  releaseDate:    isoDate().optional(),
  isExplicit:     z.boolean().default(false),
  territories:    z.array(z.string().length(2).toUpperCase()).default(['WW']),
  label:          safeText(100).optional(),
  copyrightYear:  z.number().int().min(1900).max(new Date().getFullYear() + 2).optional(),
  copyrightHolder: safeText(200).optional(),
  upc:            z.string().regex(/^\d{12,13}$/).optional(),
  primaryLanguage: z.string().length(2).default('en'),
});

const trackCreate = z.object({
  releaseId:      cuid(),
  title:          safeText(200).min(1),
  trackNumber:    z.number().int().min(1).max(200),
  isrc:           z.string()
    .regex(/^[A-Z]{2}-?[A-Z0-9]{3}-?\d{2}-?\d{5}$/, 'Invalid ISRC format')
    .optional(),
  isExplicit:     z.boolean().default(false),
  composers:      z.array(safeText(100)).max(20).default([]),
  producers:      z.array(safeText(100)).max(20).default([]),
  featuredArtists: z.array(safeText(100)).max(10).default([]),
  audioFileKey:   z.string().max(500).min(1, 'Audio file required'),
});

// ── Payout ─────────────────────────────────────────────────────────────────

const bankAccountAdd = z.object({
  accountHolder: safeText(100).min(2, 'Account holder name required'),
  bankName:      safeText(60).min(2, 'Bank name required'),
  branchCode:    z.string().max(10).trim().default(''),
  accountNumber: z.string()
    .min(6, 'Account number too short')
    .max(20, 'Account number too long')
    .regex(/^\d+$/, 'Account number must be digits only')
    .trim(),
  accountType:   z.enum(['current', 'savings', 'transmission', 'credit']).default('current'),
  isDefault:     z.boolean().default(false),
});

const payoutRequest = z.object({
  amount:        money().min(100, 'Minimum payout is R100'),
  currency:      z.enum(['ZAR', 'USD']).default('ZAR'),
  method:        z.enum(['payfast', 'paystack', 'bank_transfer', 'paypal']),
  bankAccountId: cuid().optional(),
  paypalEmail:   z.string().email().optional(),
}).refine(
  (d) => d.method !== 'bank_transfer' || !!d.bankAccountId,
  { message: 'bankAccountId is required for bank transfer', path: ['bankAccountId'] }
).refine(
  (d) => d.method !== 'paypal' || !!d.paypalEmail,
  { message: 'paypalEmail is required for PayPal payouts', path: ['paypalEmail'] }
);

// ── Admin actions ──────────────────────────────────────────────────────────

const adminPayoutAction = z.object({
  requestId: cuid(),
  action:    z.enum(['approve', 'reject']),
  notes:     safeText(500).optional(),
});

const adminUserAction = z.object({
  userId: cuid(),
  action: z.enum(['suspend', 'unsuspend', 'verify', 'unverify', 'promote', 'demote']),
  reason: safeText(500).optional(),
  role:   z.enum(['artist', 'industry', 'admin', 'moderator', 'fan']).optional(),
});

const adminReleaseAction = z.object({
  releaseId: cuid(),
  action:    z.enum(['approve', 'reject', 'takedown', 'restore']),
  reason:    safeText(1000).optional(),
});

const adminEarningsUpload = z.object({
  platform:       safeText(100).min(1, 'Platform name required'),
  reportingPeriod: z.string().regex(/^\d{4}-\d{2}$/, 'Format: YYYY-MM'),
  rows:           z.array(z.object({
    isrc:         z.string().max(20),
    streams:      z.number().int().min(0),
    grossAmount:  money(),
    currency:     z.string().length(3).toUpperCase(),
  })).min(1).max(10_000),
});

// ── Social ─────────────────────────────────────────────────────────────────

const postCreate = z.object({
  body:      safeText(2000).min(1, 'Post body required'),
  mediaUrls: z.array(httpsUrl()).max(4).default([]),
  linkType:  z.enum(['beat', 'release', 'external']).optional(),
  linkItemId: cuid().optional(),
  linkUrl:   httpsUrl().optional(),
});

const commentCreate = z.object({
  postId:  cuid(),
  body:    safeText(500).min(1, 'Comment cannot be empty'),
  parentId: cuid().optional(),
});

const reportCreate = z.object({
  entityType: z.enum(['beat', 'release', 'post', 'comment', 'user', 'message']),
  entityId:   cuid(),
  reason:     z.enum([
    'spam', 'harassment', 'copyright', 'explicit', 'misinformation', 'other'
  ]),
  details:    safeText(1000).optional(),
});

// ── Messaging ──────────────────────────────────────────────────────────────

const messageSend = z.object({
  recipientId: cuid(),
  body:        safeText(2000).min(1, 'Message body required'),
  attachmentUrl: httpsUrl().optional(),
});

// ── Search / Discovery ─────────────────────────────────────────────────────

const searchQuery = z.object({
  q:       z.string().max(200).trim().min(1),
  type:    z.enum(['beats', 'artists', 'releases', 'all']).default('all'),
  genre:   z.enum(VALID_GENRES).optional(),
  page:    z.coerce.number().int().min(1).max(500).default(1),
  limit:   z.coerce.number().int().min(1).max(50).default(20),
  sort:    z.enum(['relevance', 'newest', 'popular', 'price_asc', 'price_desc']).default('relevance'),
  minBpm:  z.coerce.number().int().min(40).max(300).optional(),
  maxBpm:  z.coerce.number().int().min(40).max(300).optional(),
  minPrice: z.coerce.number().min(0).optional(),
  maxPrice: z.coerce.number().min(0).optional(),
});

// ── DMCA ───────────────────────────────────────────────────────────────────

const dmcaSubmit = z.object({
  contentType:   z.enum(['beat', 'release', 'post']),
  contentId:     cuid(),
  claimantName:  safeText(200).min(2),
  claimantEmail: z.string().email().max(254).trim().toLowerCase(),
  originalWorkUrl: httpsUrl().optional(),
  description:   safeText(5000).min(20, 'Please describe the infringement in detail'),
  goodFaith:     z.literal(true, { errorMap: () => ({ message: 'Good faith statement required' }) }),
  accurateInfo:  z.literal(true, { errorMap: () => ({ message: 'Accuracy statement required' }) }),
});

// ── Export ─────────────────────────────────────────────────────────────────

export const schemas = {
  auth: {
    register:       authRegister,
    login:          authLogin,
    magicLink:      magicLinkRequest,
  },
  artist: {
    update:         artistUpdate,
  },
  beat: {
    upload:         beatUpload,
    update:         beatUpdate,
  },
  release: {
    create:         releaseCreate,
    track:          trackCreate,
  },
  payout: {
    bankAccount:    bankAccountAdd,
    request:        payoutRequest,
  },
  admin: {
    payoutAction:   adminPayoutAction,
    userAction:     adminUserAction,
    releaseAction:  adminReleaseAction,
    earningsUpload: adminEarningsUpload,
  },
  social: {
    post:           postCreate,
    comment:        commentCreate,
    report:         reportCreate,
  },
  messaging: {
    send:           messageSend,
  },
  search:           searchQuery,
  dmca:             dmcaSubmit,
} as const;

// ── Response helper ────────────────────────────────────────────────────────

import { NextResponse } from 'next/server';
import type { ZodError } from 'zod';

/**
 * Standardised 400 response from a Zod parse failure.
 * Returns the first error message so the client can surface it directly.
 */
export function validationError(err: ZodError): NextResponse {
  const first = err.errors[0];
  const field = first?.path.join('.') || 'input';
  const message = first?.message ?? 'Validation failed';
  return NextResponse.json(
    { error: message, field, details: err.errors },
    { status: 400 }
  );
}
