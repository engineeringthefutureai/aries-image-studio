---
name: aries-studio-mcp
description: >-
  Provides runbooks and toolchain instructions for generating images using the
  Aries Studio MCP server. Use whenever the user requests generating images via
  Aries Studio, refers to Aries Image Studio, or restricts image generation to
  the Aries MCP service.
---

# Aries Studio MCP Skill

Runbook for generating and retrieving images via the Aries Image Studio MCP service.

## Core Invariants
- **No Configuration Probing**: Never inspect internal configuration files. The `aries-studio` MCP server is already mounted, active, and operational. Do not call `studio_status` routinely—only probe status if other APIs fail or are not acting as expected.
- **Invocation Method**: Aries Studio is a lazy-loaded MCP server. Never search the filesystem for an `aries` binary or CLI tool. Always invoke tools directly using the built-in `call_mcp_tool` tool with `ServerName: "aries-studio"`.
- **Direct Agent Execution**: Call `call_mcp_tool` directly in the primary session. Do not delegate to generic subagents or attempt to check MCP registration.
- **Batch Processing & Polling**: Support multiple images per prompt using `"count": N` (up to 4). Monitor all active jobs together by passing multiple IDs (up to 16) in a single `get_jobs` call.
- **Asynchronous Execution**: Image jobs take time to render on GPU cards. Always use `get_jobs` with `wait_seconds` (up to 50s) in a monitoring loop until completion.
- **Workspace Download on Demand**: Keeping images in the step directory is fine by default; only download/copy them into the workspace `images/` directory if the user or conversation context requires it.
- **Presentation**: Always render images in-place using markdown image embedding (`![alt](file://...)`) rather than plain text links, accompanied by the image's Aries Image Studio link exactly as `get_jobs` returns it (`<studio>/images/<job_id>`).

---

## Generation Profiles (Size, Quality / Steps & Aspect Ratio)

Select parameters based on the request:
- **Aspect Ratio**: Pick the aspect ratio (`"1:1"`, `"4:3"`, `"3:4"`, `"3:2"`, `"2:3"`, `"16:9"`, `"9:16"`) based on image prompt composition (e.g. landscape scenes favor `"16:9"` or `"3:2"`, portraits or tall structures favor `"3:4"` or `"9:16"`), unless explicitly specified by the user.
- **Draft Quality**: `"size": "quarter"`, `"quality": "draft"` (12 steps)
- **Default / Unspecified**: `"size": "half"`, `"quality": "standard"` (20 steps)
- **High Quality**: `"size": "full"`, `"quality": "high"` (full size, 40 steps)

---

## Toolchain & Step-by-Step Procedure

### 1. Initiate Image Generation
Call `generate_images` directly with appropriate settings based on prompt composition and the generation profile:
```json
call_mcp_tool(
  ServerName="aries-studio",
  ToolName="generate_images",
  Arguments={
    "prompt": "<detailed description>",
    "aspect": "16:9",      // Pick based on prompt composition unless explicitly specified
    "quality": "standard", // "draft" (12 steps), "standard" (20 steps), "high" (40 steps)
    "size": "half",        // "quarter" (draft), "half" (default), "full" (high quality)
    "enhance": false,      // Set true only if user prompt needs automatic expansion
    "count": 1             // 1 to 4
  }
)
```
*Record the returned job ID(s) (e.g., `01M69YTEEXB6RK5BB88J9HEKGR`).*

### 2. Monitor & Wait for Job Completion
Poll using `get_jobs` with `wait_seconds: 50` (batch up to 16 IDs at once):
```json
call_mcp_tool(
  ServerName="aries-studio",
  ToolName="get_jobs",
  Arguments={
    "ids": ["<job_id_1>", "<job_id_2>"],
    "wait_seconds": 50
  }
)
```
- If the output reports `drawing` or `queued`, repeat the call with `wait_seconds: 50` until the status is `done` or `failed`.
- When complete, the MCP tool offloads the images to:
  `<appDataDir>/brain/<conversation-id>/.system_generated/steps/<step_index>/media_<N>.jpg`

### 3. (Optional) Download / Fetch Images to Local Workspace
Only copy images from the step directory into the workspace if the conversation context or user explicitly requests local storage. Keeping them in the step directory is fine by default:
- If copying to workspace: `mkdir -p images`
- Copy the offloaded preview files from the step directory to `images/<descriptive_name>_<job_id>.jpg`.

### 4. Present Images and Studio Links
Always render images **in-place** directly in the markdown response using image tags (`![alt](file:///...)`), paired with direct Studio links (the `<studio>/images/<job_id>` links from `get_jobs`).

#### Side-by-Side Gallery Format (Recommended for batches: 2–4 images)
Use a Markdown table layout so images are displayed side-by-side with their corresponding Studio links underneath:

```markdown
### <Request Title or Prompt Summary>
**Prompt:** *<detailed prompt>*
**Aspect Ratio:** 16:9 | **Size:** Quarter | **Quality:** Draft

| Image 1 | Image 2 | Image 3 |
| :---: | :---: | :---: |
| ![Description 1](file:///path/to/media_0.jpg) | ![Description 2](file:///path/to/media_1.jpg) | ![Description 3](file:///path/to/media_2.jpg) |
| [Studio Link](<studio>/images/<job_id_1>) | [Studio Link](<studio>/images/<job_id_2>) | [Studio Link](<studio>/images/<job_id_3>) |
```

#### Single Image Format
For single image outputs, render the image in-place followed immediately by the studio link:
```markdown
![Description](file:///path/to/media_0.jpg)
[Open in Aries Image Studio](<studio>/images/<job_id>)
```
