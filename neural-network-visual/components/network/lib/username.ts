// Guardrails for player-entered leaderboard names.
// Mirrors the backend allowlist in backend/app.py (validate_username /
// sanitize_username): letters, digits, _ and - only, 1–32 characters.

export const USERNAME_MAX_LENGTH = 32;

const USERNAME_PATTERN = /^[a-zA-Z0-9_-]+$/;
const USERNAME_DISALLOWED = /[^a-zA-Z0-9_-]/g;

export function isValidUsername(raw: string): boolean {
  const username = raw.trim();
  return (
    username.length > 0 &&
    username.length <= USERNAME_MAX_LENGTH &&
    USERNAME_PATTERN.test(username)
  );
}

// Display-side defence: entries stored before validation existed may still
// contain markup or control characters — strip them before rendering.
export function sanitizeUsername(raw: string): string {
  return raw.trim().replace(USERNAME_DISALLOWED, "").slice(0, USERNAME_MAX_LENGTH);
}
