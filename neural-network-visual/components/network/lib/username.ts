// Guardrails for player-entered leaderboard names.
// Mirrors the backend rules in backend/app.py (validate_username /
// sanitize_username):
//   1. Allowlist — letters, digits, _ and - only, 1–32 characters.
//   2. Profanity blocklist — checked on a normalized "letter skeleton"
//      (lowercase, digit look-alikes reversed, separators removed, repeated
//      letters collapsed) so "Sh1t", "f-u-c-k" and "fuuuck" are all caught.

export const USERNAME_MAX_LENGTH = 32;

const USERNAME_PATTERN = /^[a-zA-Z0-9_-]+$/;
const USERNAME_DISALLOWED = /[^a-zA-Z0-9_-]/g;

// Words never allowed in a leaderboard name. Must stay in sync with
// PROFANITY_BLOCKLIST in backend/app.py. Mild-but-ambiguous words ("ass",
// "hell", "damn") are deliberately left out: substring matching would wrongly
// reject names like "class" or "shell". Words that shrink to ~3 letters when
// repeats are collapsed ("piss" → "pis") are excluded for the same reason —
// they'd match unrelated names like "Pisa".
const PROFANITY_BLOCKLIST = [
  "fuck", "shit", "bitch", "bastard", "cunt", "whore", "slut",
  "wanker", "twat", "bullshit", "dickhead", "asshole",
  "arsehole", "jackass", "dumbass",
  // slurs
  "nigger", "nigga", "faggot", "chink", "kike", "tranny",
  "wetback", "towelhead", "retard",
  // crude anatomy/harassment terms
  "penis", "dildo",
];

// Reverse common letter/digit look-alikes before matching ("sh1t" → "shit").
const LEET_LOOKALIKES: Record<string, string> = {
  "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
};

function normalizeForProfanityCheck(name: string): string {
  const lowered = name.toLowerCase().replace(/[0134578]/g, (d) => LEET_LOOKALIKES[d]);
  const withoutSeparators = lowered.replace(/[-_]/g, "");
  return withoutSeparators.replace(/(.)\1+/g, "$1");
}

// Pre-collapse the list too, so needles match the collapsed haystack
// (e.g. "asshole" is searched for as "ashole" once repeats are collapsed).
const NORMALIZED_BLOCKLIST = Array.from(new Set(PROFANITY_BLOCKLIST.map(normalizeForProfanityCheck)));

export function isValidUsername(raw: string): boolean {
  const username = raw.trim();
  return (
    username.length > 0 &&
    username.length <= USERNAME_MAX_LENGTH &&
    USERNAME_PATTERN.test(username)
  );
}

// Submission-side profanity check; separate from isValidUsername so the UI can
// show a specific message instead of the generic character rules.
export function containsProfanity(raw: string): boolean {
  const skeleton = normalizeForProfanityCheck(raw);
  return NORMALIZED_BLOCKLIST.some((word) => skeleton.includes(word));
}

// Display-side defence: entries stored before validation existed may still
// contain markup or control characters — strip them before rendering. Names
// that trip the profanity filter are masked rather than shown.
export function sanitizeUsername(raw: string): string {
  const cleaned = raw.trim().replace(USERNAME_DISALLOWED, "").slice(0, USERNAME_MAX_LENGTH);
  if (containsProfanity(cleaned)) return "***";
  return cleaned;
}
