// src/app/api/creator/memberships/route.ts
// Fan membership checkout via Paystack (replaces PayFast form-POST).
// On charge.success → /api/creator/memberships/notify activates access.

export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server';
import { requireAuth } from '@/lib/auth';
import prisma from '@/lib/prisma';
import { initializeTransaction, generateReference } from '@/lib/paystack';
import { cancelMembership } from '@/lib/creator';

// GET — list caller's active memberships
export async function GET() {
  try {
    const user = await requireAuth();
    if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const memberships = await prisma.creatorMembership.findMany({
      where: { userId: user.id, status: 'active' },
      include: {
        tier:   true,
        artist: { select: { id: true, name: true, slug: true, photoUrl: true } },
      },
      orderBy: { createdAt: 'desc' },
    });

    return NextResponse.json({ memberships });
  } catch (err: any) {
    console.error('[creator/memberships] GET error:', err?.message);
    return NextResponse.json({ memberships: [] });
  }
}

// POST — initiate Paystack checkout for a membership tier
export async function POST(req: NextRequest) {
  try {
    const user = await requireAuth();
    if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const { tierId, artistId, billingInterval } = await req.json();
    if (!tierId)   return NextResponse.json({ error: 'tierId required' }, { status: 400 });
    if (!artistId) return NextResponse.json({ error: 'artistId required' }, { status: 400 });

    const tier = await prisma.creatorSubscriptionTier.findUnique({
      where: { id: tierId },
      include: { artist: { include: { user: true } } },
    });
    if (!tier?.isActive)              return NextResponse.json({ error: 'Tier not found or inactive' }, { status: 404 });
    if (tier.artistId !== artistId)   return NextResponse.json({ error: 'Tier/artist mismatch' }, { status: 400 });
    if (tier.artist.userId === user.id) return NextResponse.json({ error: 'Cannot subscribe to your own tier' }, { status: 400 });

    const existing = await prisma.creatorMembership.findFirst({ where: { userId: user.id, tierId, status: 'active' } });
    if (existing) return NextResponse.json({ error: 'Already subscribed to this tier' }, { status: 409 });

    const interval = billingInterval || 'monthly';
    const amount   = interval === 'yearly' ? tier.price * 12 * 0.9 : tier.price;

    const membership = await prisma.creatorMembership.create({
      data: {
        userId:          user.id,
        tierId,
        artistId:        tier.artistId,
        status:          'pending',
        billingInterval: interval,
        expiresAt:       new Date(),
      },
    });

    const appUrl    = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
    const reference = generateReference('MEM');

    const result = await initializeTransaction({
      email:       user.email,
      amountZAR:   Number(amount),
      reference,
      callbackUrl: `${appUrl}/artist/${tier.artist.slug}?membership=success`,
      metadata: {
        membershipId: membership.id,
        tierId,
        artistId:     tier.artistId,
        interval,
        userId:       user.id,
        type:         'membership',
      },
    });

    // Store reference so notify webhook can find the membership
    await prisma.creatorMembership.update({
      where: { id: membership.id },
      data:  { paystackReference: reference },
    });

    return NextResponse.json({ authorizationUrl: result.authorizationUrl, method: 'paystack' }, { status: 201 });
  } catch (err: any) {
    console.error('[creator/memberships] POST error:', err?.message);
    return NextResponse.json({ error: err?.message || 'Failed to initiate checkout' }, { status: 503 });
  }
}

// DELETE — cancel membership
export async function DELETE(req: NextRequest) {
  try {
    const user = await requireAuth();
    if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const { searchParams } = new URL(req.url);
    const tierId = searchParams.get('tierId');
    if (!tierId) return NextResponse.json({ error: 'tierId required' }, { status: 400 });

    const membership = await prisma.creatorMembership.findFirst({
      where: { userId: user.id, tierId, status: 'active' },
    });
    if (!membership) return NextResponse.json({ error: 'No active membership found' }, { status: 404 });

    await cancelMembership(membership.id);
    return NextResponse.json({ ok: true });
  } catch (err: any) {
    console.error('[creator/memberships] DELETE error:', err?.message);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}
