import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import axios from "axios";
import dotenv from "dotenv";

dotenv.config();
const CORE_URL = process.env.CORE_URL || "http://127.0.0.1:8000/v1/widget_text";
const server = new McpServer({ name: "dqs-mcp", version: "1.0.0" });

server.registerTool("dqs.health",
  { title: "Health", description: "status & endpoints", inputSchema: {} },
  async () => ({ content: [{ type: "text", text: JSON.stringify({ status:"ok", core: CORE_URL }, null, 2) }] })
);

server.registerTool("dqs.ask.core",
  { title: "Ask DQS Core", description: "Send a natural-language question to the deterministic core", inputSchema: { query: z.string() } },
  async ({ query }) => {
    const r = await axios.post(CORE_URL, { query }, { headers: { "Content-Type": "application/json" } });
    return { content: [{ type: "text", text: JSON.stringify({ query, answer_text: r.data?.answer_text ?? null }, null, 2) }] };
  }
);

const transport = new StdioServerTransport();
server.connect(transport).then(() => console.error("DQS MCP on stdio (core:", CORE_URL, ")"));
