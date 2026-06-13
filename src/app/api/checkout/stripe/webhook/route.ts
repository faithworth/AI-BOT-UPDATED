/**
 * POST /api/checkout/stripe/webhook
 *
 * Phase 12 — DEPRECATED. Stripe webhook removed from Vuka.
 * All payments processed via Paystack webhook at /api/checkout/paystack/webhook
 *
 * Returns 200 so any accidental Stripe webhook deliveries don't retry forever.
 */

export const dynamic = 'force-dynamic';
import { NextResponse } from 'next/server';

export async function POST() {
  return NextResponse.json({ received: true, note: 'Stripe removed — use Paystack' });
}
