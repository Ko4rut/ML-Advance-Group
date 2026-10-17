const API_BASE_URL = (
  process.env.NEXT_PUBLIC_XIANGQI_API_URL || "http://localhost:8000/api/v1"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Every endpoint uses one transport so network errors and HTTP errors behave consistently. */
export async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      0,
      "Không kết nối được FastAPI. Hãy chạy backend tại localhost:8000 và kiểm tra CORS.",
    );
  }

  if (!response.ok) {
    let detail: unknown = null;

    try {
      const payload: unknown = await response.json();
      if (payload && typeof payload === "object" && "detail" in payload) {
        detail = payload.detail;
      }
    } catch {
      // Backends may respond with non-JSON error bodies.
    }

    if (typeof detail === "string") {
      throw new ApiError(response.status, detail);
    }

    throw new ApiError(response.status, `API lỗi HTTP ${response.status}`);
  }

  return (await response.json()) as T;
}

export function postJson<T>(path: string, body: object): Promise<T> {
  return requestJson<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

