export function fetchWithTimeout(
  input: RequestInfo | URL,
  init?: RequestInit,
  ms = 15_000,
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return fetch(input, { ...init, signal: controller.signal }).finally(() => clearTimeout(id));
}
