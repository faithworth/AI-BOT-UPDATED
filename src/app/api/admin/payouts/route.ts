/**
 * GET  /api/admin/payouts?status=pending|approved|paid|rejected|all&page=1
 * POST /api/admin/payouts { requestId, action: approve|reject|mark_paid, notes?, reference? }
 */
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import prisma from '@/lib/prisma';
import { auditLog } from '@/lib/audit';
import {
  sendPayoutApproved,
  sendPayoutProcessed,
  sendPayoutFailed,
} from '@/lib/emails';

const APP_URL = () => process.env.NEXT_PUBLIC_APP_URL || 'https://vuka.co.za';

export async function GET(req: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { searchParams } = new URL(req.url);
  const status = searchParams.get('status') || 'pending';
  const page   = Math.max(1, parseInt(searchParams.get('page') || '1'));
  const limit  = 50;

  try {
    const where = status === 'all' ? {} : { status };
    const [requests, total] = await Promise.all([
      prisma.payoutRequest.findMany({
        where,
        include: {
          artist: {
            select: {
              id: true, name: true, slug: true,
              user: { select: { email: true } },
            },
          },
          bankAccount: {
            select: {
              bankName: true, accountHolder: true,
              maskedNumber: true, branchCode: true, accountType: true,
            },
          },
        },
        orderBy: { createdAt: 'asc' },
        skip:  (page - 1) * limit,
        take:  limit,
      }),
      prisma.payoutRequest.count({ where }),
    ]);

    const [pendingAgg, paidAgg] = await Promise.all([
      prisma.payoutRequest.aggregate({
        where: { status: 'pending' },
        _sum: { amount: true },
        _count: true,
      }),
      prisma.payoutRequest.aggregate({
        where: { status: 'paid' },
        _sum: { amount: true },
        _count: true,
      }),
    ]);

    return NextResponse.json({
      requests,
      total,
      page,
      pages: Math.ceil(total / limit),
      summary: {
        pendingCount:  pendingAgg._count,
        pendingAmount: pendingAgg._sum.amount || 0,
        paidCount:     paidAgg._count,
        paidAmount:    paidAgg._sum.amount || 0,
      },
    });
  } catch (err) {
    console.error('[admin/payouts] GET error:', err);
    return NextResponse.json({ error: 'Database error' }, { status: 503 });
  }
}

export async function POST(req: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  try {
    const { requestId, action, notes, reference } = await req.json();
    if (!requestId || !action)
      return NextResponse.json({ error: 'requestId and action required' }, { status: 400 });

    const request = await prisma.payoutRequest.findUnique({
      where: { id: requestId },
      include: { artist: { select: { name: true, user: { select: { email: true } } } } },
    });
    if (!request) return NextResponse.json({ error: 'Payout request not found' }, { status: 404 });

    switch (action) {
      case 'approve': {
        if (request.status !== 'pending')
          return NextResponse.json({ error: 'Can only approve pending requests' }, { status: 409 });
        await prisma.payoutRequest.update({
          where: { id: requestId },
          data: { status: 'approved', approvedAt: new Date(), adminNotes: notes || '' },
        });
        await auditLog.adminAction('payment.payout_approved', 'PayoutRequest', requestId, user.id, notes || '');
        // Phase 9: notify artist
        try {
          const artistEmail = request.artist?.user?.email;
          if (artistEmail) {
            await sendPayoutApproved({
              to: artistEmail,
              artistName: request.artist?.name || 'Artist',
              amount: Number(request.amount),
              currency: request.currency || 'ZAR',
              payoutMethod: request.bankAccountId ? 'Bank Transfer' : 'Paystack',
              referenceNumber: request.id,
              processingDays: 2,
              payoutsUrl: `${APP_URL()}/dashboard/payouts`,
            });
          }
        } catch (e) { console.error('[admin/payouts] approve email failed:', e); }
        return NextResponse.json({ ok: true, status: 'approved' });
      }
      case 'reject': {
        if (!['pending', 'approved'].includes(request.status))
          return NextResponse.json({ error: 'Can only reject pending or approved requests' }, { status: 409 });
        await prisma.payoutRequest.update({
          where: { id: requestId },
          data: { status: 'rejected', adminNotes: notes || 'Rejected by admin' },
        });
        await auditLog.adminAction('payment.payout_rejected', 'PayoutRequest', requestId, user.id, notes || '');
        // Phase 9: notify artist of failure
        try {
          const artistEmail = request.artist?.user?.email;
          if (artistEmail) {
            await sendPayoutFailed({
              to: artistEmail,
              artistName: request.artist?.name || 'Artist',
              amount: Number(request.amount),
              currency: request.currency || 'ZAR',
              reason: notes || 'Your payout request was rejected by admin. Please contact support for more details.',
              referenceNumber: request.id,
              payoutsUrl: `${APP_URL()}/dashboard/payouts`,
            });
          }
        } catch (e) { console.error('[admin/payouts] reject email failed:', e); }
        return NextResponse.json({ ok: true, status: 'rejected' });
      }
      case 'mark_paid': {
        if (request.status !== 'approved')
          return NextResponse.json({ error: 'Can only mark approved requests as paid' }, { status: 409 });
        await prisma.payoutRequest.update({
          where: { id: requestId },
          data: { status: 'paid', processedAt: new Date(), adminNotes: notes || '' },
        });
        // Record in ArtistPayout ledger
        await prisma.artistPayout.create({
          data: {
            artistId:    request.artistId,
            amount:      request.amount,
            currency:    request.currency,
            status:      'paid',
            method:      request.bankAccountId ? 'bank' : 'paystack',
            reference:   reference || '',
            notes:       notes || `Payout request ${requestId}`,
            processedAt: new Date(),
          },
        });
        await auditLog.adminAction(
          'payment.payout_processed',
          'PayoutRequest',
          requestId,
          user.id,
          `ref=${reference || 'none'}`,
        );
        // Phase 9: notify artist payment sent
        try {
          const artistEmail = request.artist?.user?.email;
          if (artistEmail) {
            await sendPayoutProcessed({
              to: artistEmail,
              artistName: request.artist?.name || 'Artist',
              amount: Number(request.amount),
              currency: request.currency || 'ZAR',
              payoutMethod: request.bankAccountId ? 'Bank Transfer' : 'Paystack',
              referenceNumber: reference || request.id,
              bankLast4: (request as any).bankAccount?.maskedNumber?.slice(-4),
              payoutsUrl: `${APP_URL()}/dashboard/payouts`,
            });
          }
        } catch (e) { console.error('[admin/payouts] mark_paid email failed:', e); }
        return NextResponse.json({ ok: true, status: 'paid' });
      }
      default:
        return NextResponse.json({ error: `Unknown action: ${action}` }, { status: 400 });
    }
  } catch (err: any) {
    console.error('[admin/payouts] POST error:', err?.message);
    return NextResponse.json({ error: 'Action failed' }, { status: 503 });
  }
}
