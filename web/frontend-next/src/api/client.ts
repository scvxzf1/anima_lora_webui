export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly payload: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

type ApiEnvelope = {
  ok?: boolean;
  error?: string;
};

export async function apiRequest<T>(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(input, {
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
    ...init,
  });
  const text = await response.text();
  let payload: unknown = {};

  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new ApiError(text, response.status, text);
    }
  }

  const envelope = payload as ApiEnvelope;
  if (!response.ok || envelope?.ok === false) {
    throw new ApiError(
      envelope?.error || `${response.status}: ${response.statusText || 'HTTP error'}`,
      response.status,
      payload,
    );
  }

  return payload as T;
}
