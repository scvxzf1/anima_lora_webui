export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly payload: unknown,
  ) {
    super(message);
    this.name = "ApiError";
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
  const headers = new Headers(init?.headers);
  if (!headers.has("Content-Type") && !(init?.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  let response: Response;
  let text: string;
  try {
    response = await fetch(input, { ...init, headers });
    text = await response.text();
  } catch (error) {
    if (
      init?.signal?.aborted ||
      !init?.method ||
      /^(GET|HEAD)$/i.test(init.method)
    )
      throw error;
    throw new ApiError(
      "连接中断，操作结果尚未确认。请先刷新对应记录核对服务器状态，再决定是否重试。",
      0,
      null,
    );
  }
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
      envelope?.error ||
        `${response.status}: ${response.statusText || "HTTP error"}`,
      response.status,
      payload,
    );
  }

  return payload as T;
}
