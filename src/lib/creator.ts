// ============================================================
// PHASE 2 — src/lib/creator.ts
// Creator Economy: subscriptions, memberships, storefronts,
// exclusive content, analytics, revenue tracking.
// ============================================================

import prisma, { queryRaw, executeRaw } from './prisma';
import { Prisma } from '@prisma/client';
import { platformFee as calcFee, artistNet as calcNet } from './plans';

function getPeriod(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
}

// ── Subscription Tier Management ──────────────────────────────

export async function getArtistTiers(artistId: string) {
  // Raw query — perks is jsonb in DB but Prisma schema says String[]; findMany crashes.
  const rows = await queryRaw(
    `SELECT t.*, COUNT(m.id)::int AS "membershipCount"
       FROM "CreatorSubscriptionTier" t
       LEFT JOIN "CreatorMembership" m ON m."tierId" = t.id
      WHERE t."artistId" = $1 AND t."isActive" = true
      GROUP BY t.id
      ORDER BY t."createdAt" ASC`,
    artistId,
  );
  return rows.map(r => ({
    ...r,
    perks: Array.isArray(r.perks) ? r.perks : (typeof r.perks === 'string' ? JSON.parse(r.perks) : []),
    _count: { memberships: r.membershipCount ?? 0 },
  }));
}

export async function createTier(
  artistId: string,
  data: {
    name: string;
    description?: string;
    priceMonthly: number;
    priceYearly?: number;
    currency?: string;
    perks?: { icon: string; title: string; description: string }[] | string[];
    maxSubscribers?: number;
    sortOrder?: number;
  }
) {
  // Normalise perks to plain strings
  const perksStrings: string[] = (data.perks || []).map((p) =>
    typeof p === 'string' ? p : `${(p as any).icon ?? ''} ${(p as any).title ?? ''}: ${(p as any).description ?? ''}`.trim()
  ).filter(Boolean);

  // Use Prisma.$executeRaw to bypass the String[] vs jsonb mismatch.
  // The DB column is jsonb (from an older migration) but Prisma schema says String[].
  // We insert directly so the json array lands correctly.
  const id       = require('crypto').randomUUID().replace(/-/g, '').slice(0, 25);
  const now      = new Date().toISOString();
  const price    = data.priceMonthly;
  const currency = data.currency || 'ZAR';
  const interval = 'monthly';
  const desc     = data.description || '';
  const perksJson = JSON.stringify(perksStrings);

  await executeRaw(
    `INSERT INTO "CreatorSubscriptionTier"
       (id, "artistId", name, "priceMonthly", currency, description, perks, "isActive", "createdAt", "updatedAt")
     VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, true, $8::timestamptz, $8::timestamptz)`,
    id, artistId, data.name, price, currency, desc, perksJson, now,
  );

  // Raw query — Prisma findUnique crashes reading perks (jsonb vs String[])
  const rows = await queryRaw(
    `SELECT * FROM "CreatorSubscriptionTier" WHERE id = $1 LIMIT 1`, id,
  );
  if (!rows[0]) return null;
  const r = rows[0];
  return { ...r, perks: Array.isArray(r.perks) ? r.perks : (typeof r.perks === 'string' ? JSON.parse(r.perks) : []) };
}

// ── Membership Lifecycle ──────────────────────────────────────

export async function createMembership(params: {
  userId: string;
  tierId: string;
  artistId: string;
  billingInterval: 'monthly' | 'yearly';
  paystackReference?: string;
  fanName?: string;
  fanEmail?: string;
}) {
  interface TierRow {
    id: string;
    artistId: string;
    name: string;
    priceMonthly: number;
    price?: number;
    currency: string;
    interval: string;
    description: string;
    perks: string[] | string;
    isActive: boolean;
    createdAt: Date;
  }

  const tierRows = await queryRaw<TierRow>(
    `SELECT * FROM "CreatorSubscriptionTier" WHERE id = $1 LIMIT 1`, params.tierId,
  );
  if (!tierRows[0]) throw new Error('Tier not found');
  const tierRaw = tierRows[0];
  const tier = { ...tierRaw, perks: Array.isArray(tierRaw.perks) ? tierRaw.perks : (typeof tierRaw.perks === 'string' ? JSON.parse(tierRaw.perks) : []) };

  // Fetch artist plan for accurate fee rate
  const artist = await prisma.artist.findUnique({
    where: { id: params.artistId },
    select: { planSlug: true, planExpiresAt: true },
  });

  const now = new Date();
  const periodEnd = new Date(now);
  if (params.billingInterval === 'yearly') {
    periodEnd.setFullYear(periodEnd.getFullYear() + 1);
  } else {
    periodEnd.setMonth(periodEnd.getMonth() + 1);
  }

  const amount = tier.priceMonthly ?? tier.price;
  const fee    = calcFee(amount, artist?.planSlug, artist?.planExpiresAt);
  const net    = calcNet(amount, artist?.planSlug, artist?.planExpiresAt);
  const period = getPeriod();

  return prisma.$transaction(async (tx) => {
    // Upsert membership record
    const existing = await tx.creatorMembership.findFirst({
      where: { userId: params.userId, tierId: params.tierId },
    });
    const membership = existing
      ? await tx.creatorMembership.update({
          where: { id: existing.id },
          data: { status: 'active', expiresAt: periodEnd },
        })
      : await tx.creatorMembership.create({
          data: {
            userId:    params.userId,
            tierId:    params.tierId,
            artistId:  params.artistId,
            status:    'active',
            expiresAt: periodEnd,
          },
        });

    // Record as SupportTxn so it appears in Finance > Tips tab
    const fan = await tx.user.findUnique({
      where: { id: params.userId },
      select: { email: true, name: true },
    });
    await tx.supportTxn.create({
      data: {
        fanUserId: params.userId,
        fanEmail:  fan?.email  || params.fanEmail  || '',
        fanName:   fan?.name   || params.fanName   || 'Fan',
        artistId:  params.artistId,
        amount,
        currency:  tier.currency || 'ZAR',
        tier:      tier.name,
        message:   `Fan membership: ${tier.name} (${params.billingInterval})`,
        status:    'confirmed',
      },
    });

    // Queue net payout for artist
    await tx.artistPayout.create({
      data: {
        artistId:  params.artistId,
        amount:    net,
        currency:  tier.currency || 'ZAR',
        method:    'bank',
        status:    'pending',
        notes:     `Fan membership: ${tier.name} — Vuka kept R${fee.toFixed(2)}`,
      },
    });

    // Platform RevenueRecord
    await tx.revenueRecord.create({
      data: {
        artistId:    params.artistId,
        type:        'membership',
        amount,
        platformFee: fee,
        netAmount:   net,
        period,
        currency:    tier.currency || 'ZAR',
      },
    });

    return membership;
  });
}

export async function cancelMembership(userId: string, tierId: string) {
  const membership = await prisma.creatorMembership.findFirst({ where: { userId, tierId } });
  if (!membership) throw new Error('Membership not found');
  return prisma.creatorMembership.update({
    where: { id: membership.id },
    data: { status: 'cancelled' },
  });
}

export async function renewMembership(membershipId: string, amount: number) {
  const membership = await prisma.creatorMembership.findUnique({
    where: { id: membershipId },
    include: {
      tier: true,
      artist: { select: { planSlug: true, planExpiresAt: true } },
    },
  });
  if (!membership) throw new Error('Membership not found');

  const now = new Date();
  const periodEnd = new Date(now);
  periodEnd.setMonth(periodEnd.getMonth() + 1);

  const fee    = calcFee(amount, membership.artist?.planSlug, membership.artist?.planExpiresAt);
  const net    = calcNet(amount, membership.artist?.planSlug, membership.artist?.planExpiresAt);
  const period = getPeriod();

  return prisma.$transaction(async (tx) => {
    const updated = await tx.creatorMembership.update({
      where: { id: membershipId },
      data: { status: 'active', expiresAt: periodEnd },
    });

    // Queue payout for renewal
    await tx.artistPayout.create({
      data: {
        artistId:  membership.artistId,
        amount:    net,
        currency:  membership.tier?.currency || 'ZAR',
        method:    'bank',
        status:    'pending',
        notes:     `Membership renewal: ${membership.tier?.name || 'tier'} — Vuka kept R${fee.toFixed(2)}`,
      },
    });

    // Revenue record for renewal
    await tx.revenueRecord.create({
      data: {
        artistId:    membership.artistId,
        type:        'membership',
        amount,
        platformFee: fee,
        netAmount:   net,
        period,
        currency:    membership.tier?.currency || 'ZAR',
      },
    });

    return updated;
  });
}

// ── Entitlement Check ─────────────────────────────────────────

export async function checkContentEntitlement(
  userId: string,
  contentId: string
): Promise<boolean> {
  const content = await prisma.exclusiveContent.findUnique({
    where: { id: contentId },
  });
  if (!content) return false;
  if (!content.isPublished) return false;
  if (content.isFreePreview) return true;

  const activeMembership = await prisma.creatorMembership.findFirst({
    where: {
      userId,
      artistId: content.artistId,
      status: 'active',
      expiresAt: { gte: new Date() },
    },
  });
  if (!activeMembership) return false;

  const tierIds = content.accessTierIds as string[];
  if (tierIds.length === 0) return true;

  return tierIds.includes(activeMembership.tierId);
}

// ── Creator Analytics ─────────────────────────────────────────

export async function getCreatorAnalytics(artistId: string) {
  const now = new Date();
  const thisMonth = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;

  const [
    activeMembers,
    beatSalesThisMonth,
    releaseSalesThisMonth,
    pendingPayouts,
    recentRevenue,
  ] = await Promise.all([
    prisma.creatorMembership.count({
      where: { artistId, status: 'active', expiresAt: { gte: now } },
    }),
    prisma.purchase.aggregate({
      where: {
        status: 'confirmed',
        itemType: 'beat',
        beat: { artistId },
        createdAt: { gte: new Date(now.getFullYear(), now.getMonth(), 1) },
      },
      _sum: { netAmount: true },
      _count: true,
    }),
    prisma.purchase.aggregate({
      where: {
        status: 'confirmed',
        itemType: 'release',
        release: { artistId },
        createdAt: { gte: new Date(now.getFullYear(), now.getMonth(), 1) },
      },
      _sum: { netAmount: true },
      _count: true,
    }),
    prisma.artistPayout.aggregate({
      where: { artistId, status: 'pending' },
      _sum: { amount: true },
    }),
    prisma.revenueRecord.findMany({
      where: { artistId },
      orderBy: { period: 'desc' },
      take: 12,
    }),
  ]);

  return {
    activeMembers,
    totalMemberRevenue: 0,
    beatSalesThisMonth: {
      revenue: beatSalesThisMonth._sum.netAmount || 0,
      count: beatSalesThisMonth._count,
    },
    releaseSalesThisMonth: {
      revenue: releaseSalesThisMonth._sum.netAmount || 0,
      count: releaseSalesThisMonth._count,
    },
    pendingPayouts: pendingPayouts._sum.amount || 0,
    revenueHistory: recentRevenue,
    period: thisMonth,
  };
}

// ── Revenue Record Upsert ─────────────────────────────────────

export async function upsertRevenueRecord(
  artistId: string,
  period: string,
  delta: {
    type?: string;
    amount?: number;
    platformFee?: number;
    netAmount?: number;
    purchaseId?: string;
  }
) {
  if (!delta.amount || delta.amount <= 0) return null;
  return prisma.revenueRecord.create({
    data: {
      artistId,
      period,
      type:        delta.type        || 'other',
      amount:      delta.amount      || 0,
      platformFee: delta.platformFee || 0,
      netAmount:   delta.netAmount   || 0,
      purchaseId:  delta.purchaseId,
      currency:    'ZAR',
    },
  });
}

export async function getOrCreateStorefront(artistId: string) {
  return prisma.creatorStorefront.upsert({
    where: { artistId },
    create: { artistId },
    update: {},
  });
}

// Accepts both legacy field names (heroHeadline, heroSubtext, isLive, etc.)
// and schema field names (headline, description, theme, sections, isPublic).
// Maps everything to actual schema columns before writing.
export async function updateStorefront(
  artistId: string,
  data: Record<string, unknown>
) {
  const mapped: {
    headline?: string;
    description?: string;
    theme?: string;
    sections?: Prisma.InputJsonValue;
    isPublic?: boolean;
  } = {};

  if (data.headline     !== undefined) mapped.headline    = String(data.headline);
  if (data.heroHeadline !== undefined) mapped.headline    = String(data.heroHeadline);
  if (data.description  !== undefined) mapped.description = String(data.description);
  if (data.heroSubtext  !== undefined) mapped.description = String(data.heroSubtext);
  if (data.theme        !== undefined) mapped.theme       = String(data.theme);
  if (data.accentColor  !== undefined) mapped.theme       = String(data.accentColor);
  if (data.sections     !== undefined && data.sections !== null) {
    mapped.sections = data.sections;
  }
  if (data.isPublic     !== undefined) mapped.isPublic    = Boolean(data.isPublic);
  if (data.isLive       !== undefined) mapped.isPublic    = Boolean(data.isLive);

  return prisma.creatorStorefront.upsert({
    where: { artistId },
    create: { artistId, ...mapped },
    update: mapped,
  });
}
