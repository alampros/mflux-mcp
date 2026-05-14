import { defineHarness } from '@cardor/agent-harness-kit'

export default defineHarness({
  project: {
    name: 'mflux-mcp',
    description: 'MCP server exposing the mflux image generation tool to LLM agents via the Model Context Protocol',
    docsPath: './docs',
  },

  provider: 'opencode',

  agents: {
    lead:     { instructionsPath: null },
    explorer: { instructionsPath: null, allowedPaths: ['./src', './tests', './docs', './PLAN.md'] },
    builder:  { instructionsPath: null, writablePaths: ['./src', './tests', './server.py', './pyproject.toml'] },
    reviewer: { instructionsPath: null },
    custom:   [],
  },

  database: { type: 'sqlite', path: '.harness/harness.db' },

  storage: {
    dir:    '.harness',
    tasks:  { adapter: 'local' },
    sections: {
      toolsUsed:     true,
      filesModified: true,
      result:        true,
      blockers:      true,
      nextSteps:     false,
    },
    markdownFallback: { enabled: true, path: '.harness/current.md' },
  },

  health: {
    scriptPath: './health.sh',
    required:   true,
  },

  tools: {
    mcp:     { enabled: true, port: 3742 },
    scripts: { enabled: true, outputDir: './.harness/scripts' },
  },
})
