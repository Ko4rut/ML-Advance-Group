"use client";

import { useEffect, useState } from "react";
import { getAgents } from "../api/agents";
import type { AgentId } from "../types";

function initialAvailableAgents(): Set<AgentId> {
  return new Set<AgentId>(["human"]);
}

/**
 * Fetch backend-supported agents once. The cancellation flag prevents an unmounted
 * component from updating React state when a request finishes late.
 */
export function useAgentCatalog() {
  const [available, setAvailable] = useState<Set<AgentId>>(initialAvailableAgents);
  const [connected, setConnected] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [unavailableReasons, setUnavailableReasons] = useState<Partial<Record<AgentId, string>>>({});

  useEffect(function loadCatalogEffect() {
    let cancelled = false;

    async function loadCatalog(): Promise<void> {
      try {
        const catalog = await getAgents();
        if (cancelled) {
          return;
        }

        const nextAvailable = new Set<AgentId>(["human"]);
        const reasons: Partial<Record<AgentId, string>> = {};
        for (const agent of catalog) {
          if (agent.available) {
            nextAvailable.add(agent.id as AgentId);
          } else if (agent.unavailable_reason) {
            reasons[agent.id as AgentId] = agent.unavailable_reason;
          }
        }

        setAvailable(nextAvailable);
        setUnavailableReasons(reasons);
        setConnected(true);
        setCatalogError(null);
      } catch (error: unknown) {
        if (cancelled) {
          return;
        }

        setConnected(false);
        if (error instanceof Error) {
          setCatalogError(error.message);
          return;
        }
        setCatalogError("Không tải được danh sách agent.");
      }
    }

    void loadCatalog();
    return function cancelCatalogRequest() {
      cancelled = true;
    };
  }, []);

  return { available, unavailableReasons, connected, catalogError };
}

