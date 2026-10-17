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

  useEffect(function loadCatalogEffect() {
    let cancelled = false;

    async function loadCatalog(): Promise<void> {
      try {
        const catalog = await getAgents();
        if (cancelled) {
          return;
        }

        const nextAvailable = new Set<AgentId>(["human"]);
        for (const agent of catalog) {
          if (agent.available) {
            nextAvailable.add(agent.id as AgentId);
          }
        }

        setAvailable(nextAvailable);
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

  return { available, connected, catalogError };
}

