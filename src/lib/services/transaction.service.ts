// src/lib/services/transaction.service.ts
// Centralized transaction service — all purchases, licenses, and payout records
// go through here. Uses Prisma interactive transactions so nothing partially commits.

import prisma from '@/lib/prisma';
import { createNotification, notifyArtistOfSale } from './notification.service';
import { platformFee as calcFee, artistNet as calcNet } from '@/lib/plans';

// ── Types ────────────────────────────────────────────────────

export interface BeatPurchaseInput {
  buyerUserId: string | null;  // null for guest checkouts (Purchase.userId is String? in schema)
  buyerEmail: string;
  buyerName: string;
  artistId: string;
  artistUserId: string;
  beatId: string;
  beatTitle: string;
  amount: number; // gross paid by buyer (ZAR)
  currency?: string;
  licenseType?: string; // basic | exclusive | unlimited
  paystackReference?: string; // maps to Purchase.paystackReference in schema
  downloadToken?: string;
}

export interface ReleasePurchaseInput {
  buyerUserId: string | null;  // null for guest checkouts
  buyerEmail: string;
  buyerName: string;
  artistId: string;
  artistUserId: string;
  releaseId: string;
  releaseTitle: string;
  amount: number;
  currency?: string;
  paystackReference?: string;
  downloadToken?: string;
}

// ── Beat Purchase ─────────────────────────────────────────────
// Called from the Paystack webhook after payment is confirmed.
// UPDATES an existing pending Purchase rather than creating a duplicate.

export async function processBeatPurchase(input: BeatPurchaseInput) {
  const {
    buyerUserId, buyerEmail, buyerName, artistId, artistUserId,
    beatId, beatTitle, amount, currency = 'ZAR',
    licenseType = 'basic', paystackReference, downloadToken,
  } = input;

  // Resolve artist plan for correct fee rate (respects expiry)
  const artist = await prisma.artist.findUnique({ where: { id: artistId }, select: { planSlug: true, planExpiresAt: true } });
  const platformFee = calcFee(amount, artist?.planSlug, artist?.planExpiresAt);
  const artistNet   = calcNet(amount, artist?.planSlug, artist?.planExpiresAt);

  return prisma.$transaction(async (tx) => {
    // 1. Find or create the purchase record.
    //    The checkout/initiate route already created a pending row; we update it.
    //    If for any reason it doesn't exist we upsert safely.
    const existing = await tx.purchase.findFirst({
      where: { beatId, ...(buyerUserId ? { userId: buyerUserId } : { buyerEmail }), status: 'pending' },
      select: { id: true },
    });

    const purchase = existing
      ? await tx.purchase.update({
          where: { id: existing.id },
          data: {
            status:            'confirmed',
            paystackReference: paystackReference ?? null,
            platformFee,
            netAmount:         artistNet,
          },
        })
      : await tx.purchase.create({
          data: {
            userId:            buyerUserId,
            buyerEmail,
            buyerName,
            itemType:          'beat',
            beatId,
            amount,
            currency,
            licenseType,
            paystackReference: paystackReference ?? null,
            downloadToken:     downloadToken ?? undefined, // schema has @default(cuid())
            platformFee,
            netAmount:         artistNet,
            status:            'confirmed',
          },
        });

    // 2. Increment beat sales counter
    await tx.beat.update({
      where: { id: beatId },
      data:  { sales: { increment: 1 } },
    });

    // 3. Record artist payout (pending — batched to actual payout run)
    await tx.artistPayout.create({
      data: {
        artistId,
        amount:  artistNet,
        method:  'platform',
        status:  'pending',
        notes:   `Beat sale: ${beatTitle} (purchase ${purchase.id})`,
      },
    });

    return { purchase, artistNet, platformFee };
  }).then(async (result) => {
    // Fire notifications outside the transaction (non-critical — never block the commit)
    try {
      await notifyArtistOfSale(artistUserId, buyerName, beatTitle, amount, currency);
    } catch (err) {
      console.error('[transaction] notifyArtistOfSale failed (non-fatal):', err);
    }
    return result;
  });
}

// ── Release Purchase ──────────────────────────────────────────

export async function processReleasePurchase(input: ReleasePurchaseInput) {
  const {
    buyerUserId, buyerEmail, buyerName, artistId, artistUserId,
    releaseId, releaseTitle, amount, currency = 'ZAR',
    paystackReference, downloadToken,
  } = input;

  const artist = await prisma.artist.findUnique({ where: { id: artistId }, select: { planSlug: true, planExpiresAt: true } });
  const platformFee = calcFee(amount, artist?.planSlug, artist?.planExpiresAt);
  const artistNet   = calcNet(amount, artist?.planSlug, artist?.planExpiresAt);

  return prisma.$transaction(async (tx) => {
    const existing = await tx.purchase.findFirst({
      where: { releaseId, ...(buyerUserId ? { userId: buyerUserId } : { buyerEmail }), status: 'pending' },
      select: { id: true },
    });

    const purchase = existing
      ? await tx.purchase.update({
          where: { id: existing.id },
          data: {
            status:            'confirmed',
            paystackReference: paystackReference ?? null,
            platformFee,
            netAmount:         artistNet,
          },
        })
      : await tx.purchase.create({
          data: {
            userId:            buyerUserId,
            buyerEmail,
            buyerName,
            itemType:          'release',
            releaseId,
            amount,
            currency,
            paystackReference: paystackReference ?? null,
            downloadToken:     downloadToken ?? undefined,
            platformFee,
            netAmount:         artistNet,
            status:            'confirmed',
          },
        });

    await tx.release.update({
      where: { id: releaseId },
      data:  { sales: { increment: 1 } },
    });

    await tx.artistPayout.create({
      data: {
        artistId,
        amount:  artistNet,
        method:  'platform',
        status:  'pending',
        notes:   `Release sale: ${releaseTitle} (purchase ${purchase.id})`,
      },
    });

    return { purchase, artistNet, platformFee };
  }).then(async (result) => {
    try {
      await notifyArtistOfSale(artistUserId, buyerName, releaseTitle, amount, currency);
    } catch (err) {
      console.error('[transaction] notifyArtistOfSale failed (non-fatal):', err);
    }
    return result;
  });
}

// ── Verify buyer hasn't already purchased ────────────────────

export async function hasPurchased(
  userId: string,
  beatId?: string,
  releaseId?: string
): Promise<boolean> {
  const where: Record<string, unknown> = { userId, status: 'confirmed' };
  if (beatId)    where.beatId    = beatId;
  if (releaseId) where.releaseId = releaseId;

  const existing = await prisma.purchase.findFirst({ where });
  return !!existing;
}

// ── Get a user's purchase download entitlement ───────────────

export async function getDownloadEntitlement(
  userId: string,
  token: string
): Promise<{ valid: boolean; beatId?: string; releaseId?: string }> {
  const purchase = await prisma.purchase.findFirst({
    where:  { userId, downloadToken: token, status: 'confirmed' },
    select: { beatId: true, releaseId: true },
  });

  if (!purchase) return { valid: false };
  return {
    valid:     true,
    beatId:    purchase.beatId    ?? undefined,
    releaseId: purchase.releaseId ?? undefined,
  };
}
