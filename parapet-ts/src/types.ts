// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

export interface ParapetConfig {
  configPath: string;
  port?: number;
  extraHosts?: string[];
  autoStart?: boolean;
}

export interface SessionOptions {
  userId?: string;
  role?: string;
}

export interface TrustSpan {
  readonly start: number;
  readonly end: number;
  readonly source: string;
}

export interface TransportOptions {
  port: number;
  interceptedHosts?: string[];
  timeoutMs?: number;
}
