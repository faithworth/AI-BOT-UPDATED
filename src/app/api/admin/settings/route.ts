/**
 * VUKA — Admin Settings (Phase 5 upgraded)
 *
 * GET  /api/admin/settings  — returns full structured settings including plans,
 *                             platforms, genres, flags, landing content, payouts
 * POST /api/admin/settings  — handles both:
 *   { key, value }               — legacy single key-value upsert
 *   { section, data }            — batch section save (plans, payouts, flags, etc.)
 */
export const dynamic = 'force-dynamic';
import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import prisma from '@/lib/prisma';
import { auditLog } from '@/lib/audit';
import { PLANS } from '@/lib/plans';

const DEFAULTS: Record<string, unknown> = {
  min_payout_zar:         150,
  payout_processing_days: 3,
  registrations_open:     true,
  distributions_open:     true,
  maintenance_mode:       false,
  feature_beat_store:     true,
  feature_video_dist:     false,
  feature_fan_tips:       true,
  platform_fee_pct:       15, // Free plan default — see src/lib/plans.ts
};

const DEFAULT_GENRES = [
  'Amapiano','Afrobeats','Gqom','Hip-Hop','Trap','R&B',
  'Drill','House','Kwaito','Gospel','Jazz','Pop','Electronic',
  'Reggae','Dancehall','Soul','Afro-House',
];

const DEFAULT_PLATFORMS = [
  { id:'spotify',       name:'Spotify',       slug:'spotify',       avgDeliveryDays:3,  isActive:true },
  { id:'apple-music',   name:'Apple Music',   slug:'apple-music',   avgDeliveryDays:5,  isActive:true },
  { id:'youtube-music', name:'YouTube Music', slug:'youtube-music', avgDeliveryDays:7,  isActive:true },
  { id:'boomplay',      name:'Boomplay',      slug:'boomplay',      avgDeliveryDays:7,  isActive:true },
  { id:'audiomack',     name:'Audiomack',     slug:'audiomack',     avgDeliveryDays:3,  isActive:true },
  { id:'deezer',        name:'Deezer',        slug:'deezer',        avgDeliveryDays:7,  isActive:true },
  { id:'tidal',         name:'Tidal',         slug:'tidal',         avgDeliveryDays:10, isActive:true },
  { id:'amazon-music',  name:'Amazon Music',  slug:'amazon-music',  avgDeliveryDays:7,  isActive:true },
  { id:'soundcloud',    name:'SoundCloud',    slug:'soundcloud',    avgDeliveryDays:3,  isActive:true },
  { id:'tiktok-music',  name:'TikTok Music',  slug:'tiktok-music',  avgDeliveryDays:7,  isActive:true },
  { id:'mdundo',        name:'Mdundo',        slug:'mdundo',        avgDeliveryDays:7,  isActive:true },
  { id:'pandora',       name:'Pandora',       slug:'pandora',       avgDeliveryDays:14, isActive:false },
];

// Plans come from src/lib/plans.ts — single source of truth
const DEFAULT_PLANS = PLANS.map(p => ({
  id:              p.slug,
  slug:            p.slug,
  name:            p.name,
  priceZAR:        p.priceZAR,
  royaltyShare:    p.artistSharePct,
  platformFeePct:  p.platformFeePct,
  billingPeriod:   p.billingPeriod,
  releasesPerYear: p.releasesPerYear,
}));

async function getSettingJson(key: string): Promise<unknown | null> {
  try {
    const row = await prisma.platformSetting.findUnique({ where: { key } });
    return row?.value ?? null;
  } catch { return null; }
}

export async function GET(req: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  try {
    const rows = await prisma.platformSetting.findMany({ orderBy: { key: 'asc' } });
    const kv: Record<string, unknown> = { ...DEFAULTS };
    for (const row of rows) kv[row.key] = row.value;

    // Build structured response
    const plans    = (await getSettingJson('structured_plans') as any[]) || DEFAULT_PLANS;
    const platforms = (await getSettingJson('structured_platforms') as any[]) || DEFAULT_PLATFORMS;
    const genres   = (await getSettingJson('structured_genres') as string[]) || DEFAULT_GENRES;
    const landing  = (await getSettingJson('structured_landing') as any) || {
      heroHeadline: 'Release. Distribute. Earn. Own.',
      heroSubtext: 'The independent music platform built for African artists.',
    };

    return NextResponse.json({
      settings: kv,
      plans,
      platforms,
      genres,
      landing,
      flags: {
        enableRegistration:    kv['registrations_open'] as boolean ?? true,
        enableDistribution:    kv['distributions_open'] as boolean ?? true,
        enableBeatStore:       kv['feature_beat_store'] as boolean ?? true,
        enableVideoUpload:     kv['feature_video_dist'] as boolean ?? false,
        enableMemberships:     kv['feature_fan_tips']   as boolean ?? true,
        enableMaintenanceMode: kv['maintenance_mode']   as boolean ?? false,
      },
      payouts: {
        minPayoutAmount:       kv['min_payout_zar']          ?? 100,
        payoutProcessingDays:  kv['payout_processing_days']  ?? 3,
        paystackPayoutEmail:   kv['paystack_payout_email']   ?? '',
      },
    });
  } catch (err) {
    console.error('[admin/settings] GET error:', err);
    return NextResponse.json({ error: 'Database error' }, { status: 503 });
  }
}

export async function POST(req: NextRequest) {
  const user = await requireAdmin();
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  try {
    const body = await req.json();
    const { section, data, key, value } = body;

    // Section-based saves from admin settings page
    if (section) {
      const sectionKeyMap: Record<string, string> = {
        plans:            'structured_plans',
        platforms:        'structured_platforms',
        genres:           'structured_genres',
        landing:          'structured_landing',
        toggle_platform:  'structured_platforms',
      };

      if (section === 'flags' && data) {
        // Save each flag as individual key
        const flagKeyMap: Record<string, string> = {
          enableRegistration:    'registrations_open',
          enableDistribution:    'distributions_open',
          enableBeatStore:       'feature_beat_store',
          enableVideoUpload:     'feature_video_dist',
          enableMemberships:     'feature_fan_tips',
          enableMaintenanceMode: 'maintenance_mode',
        };
        for (const [flag, setting] of Object.entries(flagKeyMap)) {
          if (flag in data) {
            await prisma.platformSetting.upsert({
              where: { key: setting },
              update: { value: data[flag], updatedBy: user.id },
              create: { key: setting, value: data[flag], updatedBy: user.id },
            });
          }
        }
        await auditLog.adminAction('admin.flags_updated', 'PlatformSetting', 'flags', user.id, '');
        return NextResponse.json({ ok: true });
      }

      if (section === 'payouts' && data) {
        const payoutKeyMap: Record<string, string> = {
          minPayoutAmount:      'min_payout_zar',
          payoutProcessingDays: 'payout_processing_days',
          paystackPayoutEmail:  'paystack_payout_email',
        };
        for (const [field, setting] of Object.entries(payoutKeyMap)) {
          if (field in data) {
            await prisma.platformSetting.upsert({
              where: { key: setting },
              update: { value: data[field], updatedBy: user.id },
              create: { key: setting, value: data[field], updatedBy: user.id },
            });
          }
        }
        await auditLog.adminAction('admin.payout_settings_updated', 'PlatformSetting', 'payouts', user.id, '');
        return NextResponse.json({ ok: true });
      }

      if (section === 'toggle_platform' && data?.id) {
        const existing = await getSettingJson('structured_platforms') as any[] || DEFAULT_PLATFORMS;
        const updated = existing.map((p: any) => p.id === data.id ? { ...p, isActive: data.isActive } : p);
        await prisma.platformSetting.upsert({
          where: { key: 'structured_platforms' },
          update: { value: updated as any, updatedBy: user.id },
          create: { key: 'structured_platforms', value: updated as any, updatedBy: user.id },
        });
        return NextResponse.json({ ok: true });
      }

      const storageKey = sectionKeyMap[section];
      if (storageKey && data !== undefined) {
        await prisma.platformSetting.upsert({
          where: { key: storageKey },
          update: { value: data as any, updatedBy: user.id },
          create: { key: storageKey, value: data as any, updatedBy: user.id },
        });
        await auditLog.adminAction(`admin.${section}_updated`, 'PlatformSetting', section, user.id, '');
        return NextResponse.json({ ok: true });
      }
    }

    // Legacy single key-value save
    if (key) {
      await prisma.platformSetting.upsert({
        where: { key },
        update: { value, updatedBy: user.id },
        create: { key, value, updatedBy: user.id },
      });
      await auditLog.adminAction('admin.setting_changed', 'PlatformSetting', key, user.id, `→ ${JSON.stringify(value)}`);
      return NextResponse.json({ ok: true });
    }

    return NextResponse.json({ error: 'key or section is required' }, { status: 400 });
  } catch (err: any) {
    console.error('[admin/settings] POST error:', err?.message);
    return NextResponse.json({ error: 'Failed to save setting' }, { status: 503 });
  }
}
