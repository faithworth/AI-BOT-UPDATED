import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/lib/prisma';

export async function GET(_req: NextRequest, { params }: { params: { slug: string } }) {
  const artist = await prisma.artist.findUnique({
    where: { slug: params.slug },
    select: {
      id: true, slug: true, name: true, bio: true, city: true, country: true,
      photoUrl: true, coverUrl: true, genreTags: true, socialLinks: true,
      currency: true, totalPlays: true, paystackRecipient: true,
      goals: { where: { isActive: true, OR: [{ deadline: null }, { deadline: { gt: new Date() } }] }, orderBy: { createdAt: 'desc' } },
      beats: {
        where: { isActive: true },
        select: { id: true, title: true, slug: true, artworkUrl: true, previewUrl: true, basicPrice: true, bpm: true, keySignature: true, genre: true, tags: true, waveformData: true },
        orderBy: { createdAt: 'desc' },
      },
      releases: {
        where: { isActive: true },
        select: { id: true, title: true, slug: true, artworkUrl: true, releaseType: true, price: true },
        orderBy: { createdAt: 'desc' },
      },
      supportReceived: {
        where: { isPublic: true, status: 'confirmed' },
        select: { fanName: true, tier: true, message: true, amount: true, currency: true },
        orderBy: { createdAt: 'desc' },
        take: 20,
      },
      followers: { select: { id: true } },
    },
  });
  if (!artist) return NextResponse.json({ error: 'Artist not found' }, { status: 404 });
  return NextResponse.json(artist);
}
