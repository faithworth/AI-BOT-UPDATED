-- 20260612b_paystack_migration_apply
-- Run this in the Supabase SQL editor to finish the Paystack migration.
-- All statements are idempotent — safe to re-run.

-- ── Re-apply column renames (the 20260612_paystack_migration run was
--    recorded as rolled back, so these may or may not already be applied) ──

-- Purchase: payfastPfPaymentId → paystackReference
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='Purchase' AND column_name='payfastPfPaymentId') THEN
    ALTER TABLE "Purchase" RENAME COLUMN "payfastPfPaymentId" TO "paystackReference";
  END IF;
END $$;

-- SupportTxn: payfastPaymentId → paystackReference
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='SupportTxn' AND column_name='payfastPaymentId') THEN
    ALTER TABLE "SupportTxn" RENAME COLUMN "payfastPaymentId" TO "paystackReference";
  END IF;
END $$;

-- artist_plan_subscriptions: payfastPaymentId → paystackReference
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='artist_plan_subscriptions' AND column_name='payfastPaymentId') THEN
    ALTER TABLE "artist_plan_subscriptions" RENAME COLUMN "payfastPaymentId" TO "paystackReference";
  END IF;
END $$;

-- artist_plan_subscriptions: payfastToken → paystackToken
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='artist_plan_subscriptions' AND column_name='payfastToken') THEN
    ALTER TABLE "artist_plan_subscriptions" RENAME COLUMN "payfastToken" TO "paystackToken";
  END IF;
END $$;

-- Artist: payfastMerchant → paystackRecipient
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='Artist' AND column_name='payfastMerchant') THEN
    ALTER TABLE "Artist" RENAME COLUMN "payfastMerchant" TO "paystackRecipient";
  END IF;
END $$;

-- ArtistBankAccount: payfastMerchantId → paystackAccountCode
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='ArtistBankAccount' AND column_name='payfastMerchantId') THEN
    ALTER TABLE "ArtistBankAccount" RENAME COLUMN "payfastMerchantId" TO "paystackAccountCode";
  END IF;
END $$;

-- Indexes for fast webhook lookups by reference
CREATE INDEX IF NOT EXISTS "Purchase_paystackReference_idx"   ON "Purchase"("paystackReference");
CREATE INDEX IF NOT EXISTS "SupportTxn_paystackReference_idx" ON "SupportTxn"("paystackReference");
CREATE INDEX IF NOT EXISTS "ArtistPlanSub_paystackRef_idx"    ON "artist_plan_subscriptions"("paystackReference");

-- ── New: CreatorMembership.paystackReference ──────────────────────────────
-- Needed so /api/creator/memberships/notify can look up a membership by
-- the Paystack reference stored during checkout.
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'CreatorMembership' AND column_name = 'paystackReference'
  ) THEN
    ALTER TABLE "CreatorMembership" ADD COLUMN "paystackReference" TEXT;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS "CreatorMembership_paystackReference_idx" ON "CreatorMembership"("paystackReference");

-- ── New: MarketplaceOrder.paystackReference ───────────────────────────────
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'MarketplaceOrder' AND column_name = 'paystackReference'
  ) THEN
    ALTER TABLE "MarketplaceOrder" ADD COLUMN "paystackReference" TEXT;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS "MarketplaceOrder_paystackReference_idx" ON "MarketplaceOrder"("paystackReference");

-- ── Mark the previously-rolled-back migration as applied ──────────────────
-- (Run only after the above succeeds. This prevents `npx prisma migrate deploy`
--  from trying — and failing — to re-run 20260612_paystack_migration.)
-- UPDATE "_prisma_migrations"
--   SET finished_at = now(), rolled_back_at = NULL, applied_steps_count = 1, logs = NULL
--   WHERE migration_name = '20260612_paystack_migration';
