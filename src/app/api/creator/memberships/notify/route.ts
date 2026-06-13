// src/app/api/creator/memberships/notify/route.ts
// Paystack webhook — activates fan membership on confirmed payment.
// Reference stored in CreatorMembership.paystackReference during checkout.
//
// NOTE: Paystack only allows ONE webhook URL per account. The single
// registered URL should be /api/checkout/paystack/webhook, which dispatches
// MEM_ references to handleMembershipEvent automatically. This route is
// kept as a standalone wrapper around the same handler for backward
// compatibility and direct testing.

export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server';
import { verifyPaystackWebhook } from '@/lib/paystack';
import { handleMembershipEvent } from '@/lib/webhooks/paystack-handlers';
import { logger } from '@/lib/logger';

export async function POST(req: NextRequest) {
  const traceId   = req.headers.get('x-trace-id') ?? 'no-trace';
  const signature = req.headers.get('x-paystack-signature') ?? '';
  const rawBody   = await req.text();

  if (!verifyPaystackWebhook(rawBody, signature)) {
    logger.warn('[memberships/notify] Invalid signature', { traceId });
    return new NextResponse('Invalid signature', { status: 400 });
  }

  let event: any;
  try { event = JSON.parse(rawBody); } catch { return new NextResponse('Bad JSON', { status: 400 }); }

  if (event.event !== 'charge.success') return NextResponse.json({ ok: true });

  const reference = event.data?.reference ?? '';
  if (!reference.startsWith('MEM_')) return NextResponse.json({ ok: true });

  await handleMembershipEvent(event, traceId);
  return NextResponse.json({ ok: true });
}
