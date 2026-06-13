// ============================================================
// src/lib/payouts.ts
// Rewritten to use ONLY actual schema fields.
// ArtistPayout: amount, currency, status, method, reference, purchaseId, notes, processedAt
// PayoutRequest: amount, currency, bankAccountId, status, adminNotes, processedAt
// RevenueRecord: type, amount, platformFee, netAmount, period
// ============================================================

import prisma from './prisma';

function getPeriod(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
}

// ── Request a Payout ──────────────────────────────────────────

export async function requestPayout(params: {
  artistId: string;
  amount: number;
  currency?: string;
  bankAccountId?: string;
  method?: string;
  bankAccountRef?: string;
  bankName?: string;
  accountHolder?: string;
  paypalEmail?: string;
  splits?: unknown;
}) {
  const artist = await prisma.artist.findUnique({ where: { id: params.artistId } });
  if (!artist) throw new Error('Artist not found');

  // Derive available balance from confirmed purchases netAmount
  const [beatSales, releaseSales] = await Promise.all([
    prisma.purchase.aggregate({
      where: { status: 'confirmed', beat: { artistId: params.artistId } },
      _sum: { netAmount: true },
    }),
    prisma.purchase.aggregate({
      where: { status: 'confirmed', release: { artistId: params.artistId } },
      _sum: { netAmount: true },
    }),
  ]);

  const totalEarned = (beatSales._sum.netAmount ?? 0) + (releaseSales._sum.netAmount ?? 0);

  const alreadyRequested = await prisma.payoutRequest.aggregate({
    where: { artistId: params.artistId, status: { in: ['pending', 'approved'] } },
    _sum: { amount: true },
  });
  const available = totalEarned - (alreadyRequested._sum.amount ?? 0);

  if (params.amount > available + 0.01) {
    throw new Error(`Requested ${params.amount} exceeds available balance ${available.toFixed(2)}`);
  }

  return prisma.payoutRequest.create({
    data: {
      artistId:     params.artistId,
      amount:       params.amount,
      currency:     params.currency || 'ZAR',
      bankAccountId: params.bankAccountId,
      status:       'pending',
      adminNotes:   '',
    },
  });
}

// ── Admin: Approve Payout Request ────────────────────────────

export async function approvePayoutRequest(requestId: string, notes?: string) {
  const req = await prisma.payoutRequest.findUnique({ where: { id: requestId } });
  if (!req) throw new Error('Payout request not found');
  if (req.status !== 'pending') throw new Error(`Already ${req.status}`);

  return prisma.payoutRequest.update({
    where: { id: requestId },
    data: { status: 'approved', ...(notes ? { adminNotes: notes } : {}) },
  });
}

// ── Admin: Mark Payout Paid ───────────────────────────────────

export async function markPayoutPaid(requestId: string, reference: string) {
  return prisma.$transaction(async (tx) => {
    const req = await tx.payoutRequest.update({
      where: { id: requestId },
      data: { status: 'paid', processedAt: new Date(), adminNotes: `Ref: ${reference}` },
    });

    // Create an ArtistPayout record as the ledger entry
    await tx.artistPayout.create({
      data: {
        artistId:  req.artistId,
        amount:    req.amount,
        currency:  req.currency,
        status:    'paid',
        method:    req.bankAccountId ? 'bank' : 'paystack',
        reference,
        notes:     `PayoutRequest ${requestId}`,
        processedAt: new Date(),
      },
    });

    return req;
  });
}

// ── Admin: Reject Payout Request ─────────────────────────────

export async function rejectPayoutRequest(requestId: string, reason: string) {
  return prisma.payoutRequest.update({
    where: { id: requestId },
    data: { status: 'rejected', adminNotes: reason },
  });
}

// ── Get Reconciliation ────────────────────────────────────────

export async function getPayoutReconciliation(artistId: string) {
  const [pending, paid, failed] = await Promise.all([
    prisma.artistPayout.aggregate({ where: { artistId, status: 'pending' }, _sum: { amount: true }, _count: true }),
    prisma.artistPayout.aggregate({ where: { artistId, status: 'paid' }, _sum: { amount: true }, _count: true }),
    prisma.artistPayout.aggregate({ where: { artistId, status: 'failed' }, _sum: { amount: true }, _count: true }),
  ]);

  const requests = await prisma.payoutRequest.findMany({
    where: { artistId },
    orderBy: { createdAt: 'desc' },
    take: 20,
    include: { bankAccount: { select: { bankName: true, maskedNumber: true } } },
  });

  return {
    balance: {
      pending:    pending._sum.amount ?? 0,
      paid:       paid._sum.amount ?? 0,
      failed:     failed._sum.amount ?? 0,
    },
    counts: {
      pending: pending._count,
      paid:    paid._count,
      failed:  failed._count,
    },
    requests,
  };
}

// ── Get Payout History ────────────────────────────────────────

export async function getPayoutHistory(artistId: string) {
  return prisma.artistPayout.findMany({
    where: { artistId },
    orderBy: { createdAt: 'desc' },
    take: 50,
  });
}

// ── Retry Failed Payout ───────────────────────────────────────

export async function retryPayoutRequest(requestId: string) {
  const req = await prisma.payoutRequest.findUnique({ where: { id: requestId } });
  if (!req) throw new Error('Payout request not found');
  if (req.status !== 'rejected') throw new Error('Only rejected requests can be retried');
  return prisma.payoutRequest.update({
    where: { id: requestId },
    data: { status: 'pending', adminNotes: 'Retried by artist', processedAt: null },
  });
}

// ── Add Bank Account ──────────────────────────────────────────

export async function addBankAccount(params: {
  artistId: string;
  bankName?: string;
  accountHolder?: string;
  accountNumber?: string;
  branchCode?: string;
  accountType?: string;
  paypalEmail?: string;
  paystackAccountCode?: string;
  setAsDefault?: boolean;
}) {
  const accountNumber = params.accountNumber || '';
  const masked = accountNumber.length > 4
    ? accountNumber.slice(-4).padStart(accountNumber.length, '*')
    : accountNumber;

  if (params.setAsDefault) {
    await prisma.artistBankAccount.updateMany({
      where: { artistId: params.artistId },
      data: { isDefault: false },
    });
  }

  return prisma.artistBankAccount.create({
    data: {
      artistId:            params.artistId,
      bankName:            params.bankName      || '',
      accountHolder:       params.accountHolder || '',
      accountNumber:       accountNumber,
      maskedNumber:        masked,
      branchCode:          params.branchCode    || '',
      accountType:         params.accountType   || 'bank',
      paypalEmail:         params.paypalEmail,
      paystackAccountCode: params.paystackAccountCode,
      isDefault:           params.setAsDefault  || false,
    },
  });
}
