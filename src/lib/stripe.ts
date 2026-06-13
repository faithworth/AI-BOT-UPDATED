/**
 * DEPRECATED — Stripe removed from Vuka platform.
 *
 * All payments are handled via:
  *   - Paystack  (South Africa — ZAR, instant EFT, card)
 *   - Flutterwave (Pan-Africa — bank transfers, mobile money)
 *   - PayPal    (International artists — USD payouts)
 *
 * This file is kept as a stub so any legacy import resolves without crashing.
 * Do NOT use these functions — they are no-ops.
 */

// No-op stubs — prevent build errors in any file that still imports from here
export const stripe = null as unknown as never;

export async function createCheckoutSession(): Promise<never> {
  throw new Error('[Vuka] Stripe has been removed. Use Paystack via /api/checkout/paystack/initialize');
}

export async function createConnectAccountLink(): Promise<never> {
  throw new Error('[Vuka] Stripe Connect has been removed. Use Paystack Payouts.');
}

export async function createConnectAccount(): Promise<never> {
  throw new Error('[Vuka] Stripe Connect has been removed. Use Paystack Payouts.');
}
