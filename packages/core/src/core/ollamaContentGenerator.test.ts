/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { OllamaContentGenerator } from './ollamaContentGenerator.js';
import type { GenerateContentParameters, Tool } from '@google/genai';
import { LlmRole } from '../telemetry/llmRole.js';

describe('OllamaContentGenerator', () => {
  const mockFetch = vi.fn();
  let defaultParams: GenerateContentParameters;

  beforeEach(() => {
    global.fetch = mockFetch;
    defaultParams = {
      model: 'test-model',
      contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
    };
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize with default config', () => {
    const generator = new OllamaContentGenerator();
    // No easy way to check internals without testing methods,
    // but we can ensure it doesn't crash.
    expect(generator).toBeInstanceOf(OllamaContentGenerator);
  });

  it('should generate content correctly', async () => {
    const generator = new OllamaContentGenerator(
      'http://localhost:11434',
      'test-model',
    );
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        model: 'test-model',
        created_at: '2023-01-01T00:00:00Z',
        message: {
          role: 'assistant',
          content: 'Hi there!',
        },
        done: true,
      }),
    });

    const result = await generator.generateContent(
      defaultParams,
      'test-id',
      LlmRole.MAIN,
    );

    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:11434/api/chat',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: expect.stringContaining('"role":"user"'),
      }),
    );
    expect(result.text).toBe('Hi there!');
    expect(result.candidates?.[0]?.content?.parts?.[0]?.text).toBe('Hi there!');
  });

  it('should include Authorization header if API key is provided', async () => {
    const generator = new OllamaContentGenerator(
      'http://localhost:11434',
      'test-model',
      'my-secret-key',
    );
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        model: 'test-model',
        message: { role: 'assistant', content: 'Secret response' },
        done: true,
      }),
    });

    await generator.generateContent(defaultParams, 'test-id', LlmRole.MAIN);

    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:11434/api/chat',
      expect.objectContaining({
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer my-secret-key',
        },
      }),
    );
  });

  it('should map tools to Ollama format correctly', async () => {
    const generator = new OllamaContentGenerator();
    const tools: Tool[] = [
      {
        functionDeclarations: [
          {
            name: 'get_weather',
            description: 'Get weather',
            parametersJsonSchema: {
              type: 'object',
              properties: { location: { type: 'string' } },
            } as unknown as Record<string, unknown>,
          },
        ],
      },
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        model: 'test-model',
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              function: {
                name: 'get_weather',
                arguments: { location: 'Tokyo' },
              },
            },
          ],
        },
        done: true,
      }),
    });

    const result = await generator.generateContent(
      { ...defaultParams, config: { tools } },
      'test-id',
      LlmRole.MAIN,
    );

    const fetchBody = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(fetchBody.tools[0].type).toBe('function');
    expect(fetchBody.tools[0].function.name).toBe('get_weather');

    expect(result.functionCalls?.[0]?.name).toBe('get_weather');
    expect(result.functionCalls?.[0]?.args).toEqual({ location: 'Tokyo' });
  });

  it('should handle streaming output', async () => {
    const generator = new OllamaContentGenerator();

    // Mocking an async iterable stream for fetch body
    const mockChunks = [
      '{"model":"test","message":{"role":"assistant","content":"Hel"},"done":false}\n',
      '{"model":"test","message":{"role":"assistant","content":"lo"},"done":true}\n',
    ];

    const mockStream = {
      async *[Symbol.asyncIterator]() {
        for (const chunk of mockChunks) {
          yield new TextEncoder().encode(chunk);
        }
      },
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      body: mockStream,
    });

    const stream = await generator.generateContentStream(
      defaultParams,
      'test-id',
      LlmRole.MAIN,
    );

    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk.text);
    }

    expect(chunks).toEqual(['Hel', 'lo']);
  });
});
