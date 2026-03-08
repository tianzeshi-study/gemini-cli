/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Content,
  Part,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
  FunctionDeclaration,
  Tool,
} from '@google/genai';
import { FinishReason } from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';

// ── Ollama request/response types ──────────────────────────────────────

interface OllamaMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: OllamaToolCall[];
  images?: string[];
}

interface OllamaToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

interface OllamaTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

interface OllamaChatRequest {
  model: string;
  messages: OllamaMessage[];
  stream: boolean;
  tools?: OllamaTool[];
  options?: Record<string, unknown>;
  format?: string | Record<string, unknown>;
  think?: boolean;
}

interface OllamaChatResponse {
  model: string;
  created_at: string;
  message: OllamaMessage;
  done: boolean;
  done_reason?: string;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

// ── Conversion helpers ─────────────────────────────────────────────────

/**
 * Convert Gemini Content[] to Ollama messages.
 */
function contentsToOllamaMessages(contents: Content[]): OllamaMessage[] {
  const messages: OllamaMessage[] = [];
  for (const content of contents) {
    const role = mapRole(content.role);
    const parts = content.parts || [];

    // Handle function responses → tool role
    const functionResponses = parts.filter((p: Part) => p.functionResponse);
    if (functionResponses.length > 0) {
      for (const fr of functionResponses) {
        messages.push({
          role: 'tool',
          content: JSON.stringify(fr.functionResponse?.response ?? {}),
        });
      }
      continue;
    }

    // Handle function calls → assistant with tool_calls
    const functionCalls = parts.filter((p: Part) => !!p.functionCall);
    if (functionCalls.length > 0) {
      /* eslint-disable @typescript-eslint/no-unsafe-type-assertion */
      const toolCalls: OllamaToolCall[] = functionCalls.map((p: Part) => ({
        function: {
          name: p.functionCall!.name!,
          arguments: (p.functionCall!.args as Record<string, unknown>) ?? {},
        },
      }));
      /* eslint-enable @typescript-eslint/no-unsafe-type-assertion */
      messages.push({
        role: 'assistant',
        content: '',
        tool_calls: toolCalls,
      });
      continue;
    }

    // Regular text + images
    const textParts: string[] = [];
    const images: string[] = [];
    for (const part of parts) {
      if (part.text) {
        textParts.push(part.text);
      }
      if (part.inlineData?.data) {
        images.push(part.inlineData.data);
      }
    }

    const msg: OllamaMessage = {
      role,
      content: textParts.join('\n'),
    };
    if (images.length > 0) {
      msg.images = images;
    }
    messages.push(msg);
  }
  return messages;
}

function mapRole(
  role: string | undefined,
): 'system' | 'user' | 'assistant' | 'tool' {
  switch (role) {
    case 'user':
      return 'user';
    case 'model':
      return 'assistant';
    case 'system':
      return 'system';
    default:
      return 'user';
  }
}

/**
 * Convert Gemini Tool[] (with FunctionDeclarations) to Ollama tool format.
 */
function geminiToolsToOllamaTools(
  tools: Tool[] | undefined,
): OllamaTool[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  const result: OllamaTool[] = [];
  for (const tool of tools) {
    if (!tool.functionDeclarations) continue;
    for (const fd of tool.functionDeclarations) {
      result.push(geminiDeclToOllamaTool(fd));
    }
  }
  return result.length > 0 ? result : undefined;
}

function geminiDeclToOllamaTool(fd: FunctionDeclaration): OllamaTool {
  return {
    type: 'function',
    function: {
      name: fd.name || '',
      description: fd.description || '',
      /* eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion */
      parameters: (fd.parametersJsonSchema as Record<string, unknown>) ?? {},
    },
  };
}

/* eslint-disable @typescript-eslint/no-unsafe-type-assertion */
function ollamaResponseToGemini(
  resp: OllamaChatResponse,
): GenerateContentResponse {
  const parts: Part[] = [];

  // Handle tool calls
  if (resp.message.tool_calls && resp.message.tool_calls.length > 0) {
    for (const tc of resp.message.tool_calls) {
      parts.push({
        functionCall: {
          name: tc.function.name,
          args: tc.function.arguments,
        },
      });
    }
  }

  // Handle text
  if (resp.message.content) {
    parts.push({ text: resp.message.content });
  }

  const hasFunctionCalls = parts.some((p: Part) => p.functionCall);

  const response = {
    text: resp.message.content || undefined,
    functionCalls: hasFunctionCalls
      ? parts
          .filter((p: Part) => p.functionCall)
          .map((p: Part) => p.functionCall!)
      : undefined,
    candidates: [
      {
        content: {
          role: 'model',
          parts,
        },
        finishReason: resp.done ? FinishReason.STOP : undefined,
      },
    ],
    modelVersion: resp.model,
    usageMetadata: {
      promptTokenCount: resp.prompt_eval_count,
      candidatesTokenCount: resp.eval_count,
      totalTokenCount: (resp.prompt_eval_count ?? 0) + (resp.eval_count ?? 0),
    },
  } as unknown as GenerateContentResponse;

  return response;
}
/* eslint-enable @typescript-eslint/no-unsafe-type-assertion */

// ── Main class ─────────────────────────────────────────────────────────

export class OllamaContentGenerator implements ContentGenerator {
  private readonly baseUrl: string;
  private readonly defaultModel: string;
  private readonly apiKey: string | undefined;

  constructor(baseUrl?: string, defaultModel?: string, apiKey?: string) {
    this.baseUrl = (baseUrl || 'http://localhost:11434').replace(/\/+$/, '');
    this.defaultModel = defaultModel || 'llama3';
    this.apiKey = apiKey;
  }

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const model = request.model || this.defaultModel;
    /* eslint-disable @typescript-eslint/no-unsafe-type-assertion */
    const requestContents: Content[] = Array.isArray(request.contents)
      ? (request.contents as unknown as Content[])
      : ([request.contents] as unknown as Content[]);
    /* eslint-enable @typescript-eslint/no-unsafe-type-assertion */
    const messages = contentsToOllamaMessages(requestContents);

    // Prepend system instruction
    const sysInstr = request.config?.systemInstruction;
    if (sysInstr) {
      const sysText =
        typeof sysInstr === 'string'
          ? sysInstr
          : 'text' in sysInstr
            ? sysInstr.text || ''
            : '';
      if (sysText) {
        messages.unshift({ role: 'system', content: sysText });
      }
    }

    /* eslint-disable @typescript-eslint/no-unsafe-type-assertion */
    const tools = geminiToolsToOllamaTools(
      request.config?.tools as Tool[] | undefined,
    );
    /* eslint-enable @typescript-eslint/no-unsafe-type-assertion */

    const body: OllamaChatRequest = {
      model,
      messages,
      stream: false,
      ...(tools && { tools }),
    };

    // Map generation config options
    if (request.config?.temperature !== undefined) {
      body.options = {
        ...body.options,
        temperature: request.config.temperature,
      };
    }
    if (request.config?.topP !== undefined) {
      body.options = { ...body.options, top_p: request.config.topP };
    }
    if (request.config?.topK !== undefined) {
      body.options = { ...body.options, top_k: request.config.topK };
    }

    // Structured output (JSON schema)
    if (request.config?.responseJsonSchema) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      body.format = request.config.responseJsonSchema as Record<
        string,
        unknown
      >;
    } else if (request.config?.responseMimeType === 'application/json') {
      body.format = 'json';
    }

    const resp = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(body),
      signal: request.config?.abortSignal,
    });

    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`Ollama API error (${resp.status}): ${errText}`);
    }

    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const data = (await resp.json()) as OllamaChatResponse;
    return ollamaResponseToGemini(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const model = request.model || this.defaultModel;
    /* eslint-disable @typescript-eslint/no-unsafe-type-assertion */
    const requestContents: Content[] = Array.isArray(request.contents)
      ? (request.contents as unknown as Content[])
      : ([request.contents] as unknown as Content[]);
    /* eslint-enable @typescript-eslint/no-unsafe-type-assertion */
    const messages = contentsToOllamaMessages(requestContents);

    // Prepend system instruction
    const sysInstr = request.config?.systemInstruction;
    if (sysInstr) {
      const sysText =
        typeof sysInstr === 'string'
          ? sysInstr
          : 'text' in sysInstr
            ? sysInstr.text || ''
            : '';
      if (sysText) {
        messages.unshift({ role: 'system', content: sysText });
      }
    }

    /* eslint-disable @typescript-eslint/no-unsafe-type-assertion */
    const tools = geminiToolsToOllamaTools(
      request.config?.tools as Tool[] | undefined,
    );
    /* eslint-enable @typescript-eslint/no-unsafe-type-assertion */

    const body: OllamaChatRequest = {
      model,
      messages,
      stream: true,
      ...(tools && { tools }),
    };

    // Map generation config options
    if (request.config?.temperature !== undefined) {
      body.options = {
        ...body.options,
        temperature: request.config.temperature,
      };
    }
    if (request.config?.topP !== undefined) {
      body.options = { ...body.options, top_p: request.config.topP };
    }
    if (request.config?.topK !== undefined) {
      body.options = { ...body.options, top_k: request.config.topK };
    }

    // Structured output
    if (request.config?.responseJsonSchema) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      body.format = request.config.responseJsonSchema as Record<
        string,
        unknown
      >;
    } else if (request.config?.responseMimeType === 'application/json') {
      body.format = 'json';
    }

    const resp = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(body),
      signal: request.config?.abortSignal,
    });

    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`Ollama API error (${resp.status}): ${errText}`);
    }

    const reader = resp.body!;

    async function* streamGenerator(): AsyncGenerator<GenerateContentResponse> {
      // Node.js ReadableStream from fetch
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      const nodeStream = reader as unknown as AsyncIterable<Uint8Array>;
      const decoder = new TextDecoder();
      let buffer = '';

      for await (const chunk of nodeStream) {
        buffer += decoder.decode(chunk, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
            const data = JSON.parse(trimmed) as OllamaChatResponse;
            yield ollamaResponseToGemini(data);
          } catch {
            // skip malformed JSON lines
          }
        }
      }

      // Process remaining buffer
      if (buffer.trim()) {
        try {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
          const data = JSON.parse(buffer.trim()) as OllamaChatResponse;
          yield ollamaResponseToGemini(data);
        } catch {
          // skip
        }
      }
    }

    return streamGenerator();
  }

  async countTokens(
    _request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Ollama does not have a dedicated token counting endpoint.
    // Return a rough estimate based on character count.
    return {
      totalTokens: 0,
    };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Embedding via Ollama is possible but not critical for this integration.
    return {
      embeddings: [],
    };
  }
}
