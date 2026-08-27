// Guardrails for player-entered leaderboard names.
// Mirrors the backend rules in backend/app.py (validate_username /
// sanitize_username):
//   1. Allowlist — letters, digits, _ and - only, 1–32 characters.
//   2. Profanity blocklist — checked on a normalized "letter skeleton"
//      (lowercase, digit look-alikes reversed, separators removed) using
//      repeat-tolerant patterns, so "Sh1t", "f-u-c-k" and "fuuuck" are all
//      caught.

export const USERNAME_MAX_LENGTH = 32;

const USERNAME_PATTERN = /^[a-zA-Z0-9_-]+$/;
const USERNAME_DISALLOWED = /[^a-zA-Z0-9_-]/g;

// Words never allowed in a leaderboard name. Must stay in sync with
// PROFANITY_BLOCKLIST in backend/app.py. Mild-but-ambiguous words ("ass",
// "hell", "damn", "piss") are deliberately left out: substring matching would
// wrongly reject names like "class", "shell" or "Pisa".
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

// Lowercased, look-alike digits reversed, separators dropped. Repeated letters
// are deliberately NOT collapsed here — see PROFANITY_PATTERNS.
function normalizeForProfanityCheck(name: string): string {
  const lowered = name.toLowerCase().replace(/[0134578]/g, (d) => LEET_LOOKALIKES[d]);
  return lowered.replace(/[-_]/g, "");
}

// Match each word with every letter allowed to repeat ("fuuuck", "shiiit"),
// rather than collapsing repeats on both sides. Collapsing the needles too
// shortens them into legitimate substrings — "nigger" collapses to "niger",
// which then rejects "Nigeria" — so the repetition is expressed in the pattern
// and the name under test is left intact.
const PROFANITY_PATTERNS = PROFANITY_BLOCKLIST.map(
  (word) => new RegExp(word.split("").map((c) => `${c}+`).join("")),
);

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
  return PROFANITY_PATTERNS.some((pattern) => pattern.test(skeleton));
}

// Display-side defence: entries stored before validation existed may still
// contain markup or control characters — strip them before rendering. Names
// that trip the profanity filter are masked rather than shown.
export function sanitizeUsername(raw: string): string {
  const cleaned = raw.trim().replace(USERNAME_DISALLOWED, "").slice(0, USERNAME_MAX_LENGTH);
  if (containsProfanity(cleaned)) return "***";
  return cleaned;
}
