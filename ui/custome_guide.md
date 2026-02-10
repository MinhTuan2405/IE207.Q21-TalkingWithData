# Customization Guide - Talking to Data UI

## Quick Start

1. **Clone Open WebUI source:**
```powershell
.\scripts\setup-custom-ui.ps1
```

2. **Customize the UI:**
Navigate to `ui-custom/open-webui/` and modify files

3. **Build and run:**
```powershell
docker-compose build open-webui
docker-compose up open-webui -d
```

## Common Customizations

### 1. Change App Name and Logo

**File:** `ui-custom/open-webui/src/lib/constants.ts`
```typescript
export const APP_NAME = 'Talking to Data';
export const APP_DESCRIPTION = 'Ask questions about your data in natural language';
```

**File:** `ui-custom/open-webui/src/app.html`
```html
<title>Talking to Data</title>
```

**Replace logo:** `ui-custom/open-webui/static/favicon.png`

### 2. Customize Chat Interface

**Message Styling:** `ui-custom/open-webui/src/lib/components/chat/Messages.svelte`
```svelte
<!-- Customize message bubbles, colors, fonts -->
<div class="message-container">
  <!-- Your custom layout -->
</div>
```

**Input Box:** `ui-custom/open-webui/src/lib/components/chat/MessageInput.svelte`
```svelte
<!-- Customize input placeholder, buttons -->
<input placeholder="Ask about your data..." />
```

### 3. Add Custom Theme Colors

**File:** `ui-custom/open-webui/tailwind.config.js`
```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3b82f6',  // Your brand color
          dark: '#1e40af',
        },
      },
    },
  },
};
```

**File:** `ui-custom/open-webui/src/app.css`
```css
:root {
  --primary-color: #3b82f6;
  --secondary-color: #10b981;
}
```

### 4. Modify Sidebar/Navigation

**File:** `ui-custom/open-webui/src/lib/components/layout/Sidebar.svelte`
```svelte
<!-- Add custom menu items -->
<nav>
  <a href="/data-explorer">Data Explorer</a>
  <a href="/query-builder">Query Builder</a>
</nav>
```

### 5. Add Custom Pages/Routes

**Create:** `ui-custom/open-webui/src/routes/data-explorer/+page.svelte`
```svelte
<script>
  // Your custom page logic
</script>

<div class="container">
  <h1>Data Explorer</h1>
  <!-- Your custom UI -->
</div>
```

### 6. Customize Backend API

**Add endpoint:** `ui-custom/open-webui/backend/open_webui/routers/custom.py`
```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/api/custom/data")
async def get_custom_data():
    return {"message": "Your custom data"}
```

**Register router:** `ui-custom/open-webui/backend/open_webui/main.py`
```python
from open_webui.routers import custom
app.include_router(custom.router)
```

### 7. Add SQL Query Feature

**Create component:** `ui-custom/open-webui/src/lib/components/custom/SqlQueryPanel.svelte`
```svelte
<script>
  let query = '';
  let results = null;

  async function executeQuery() {
    const response = await fetch('/api/custom/execute-sql', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
    results = await response.json();
  }
</script>

<div class="sql-panel">
  <textarea bind:value={query} placeholder="SELECT * FROM users" />
  <button on:click={executeQuery}>Execute</button>
  {#if results}
    <div class="results">{JSON.stringify(results)}</div>
  {/if}
</div>
```

## Development Mode

For faster iteration during development:

```powershell
cd ui-custom/open-webui

# Install dependencies
npm install

# Run dev server (frontend only)
npm run dev
```

Frontend runs at: http://localhost:5173

**For backend development:**
```powershell
cd ui-custom/open-webui/backend
pip install -r requirements.txt
uvicorn open_webui.main:app --reload --port 8080
```

Backend runs at: http://localhost:8080

## File Structure

```
ui-custom/
└── open-webui/
    ├── src/                          # Frontend source (SvelteKit)
    │   ├── routes/                   # Pages and routes
    │   ├── lib/
    │   │   ├── components/           # Reusable components
    │   │   ├── stores/               # State management
    │   │   ├── styles/               # CSS/styling
    │   │   └── utils/                # Utilities
    │   └── app.html                  # HTML template
    ├── backend/                      # Python backend
    │   └── open_webui/
    │       ├── routers/              # API routes
    │       ├── models/               # Database models
    │       └── main.py               # FastAPI app
    ├── static/                       # Static assets (images, fonts)
    ├── tailwind.config.js            # Tailwind configuration
    └── package.json                  # Node dependencies
```

## Tips

1. **Hot reload:** Use `npm run dev` for frontend changes
2. **Component reuse:** Check existing components before creating new ones
3. **Styling:** Use Tailwind CSS classes for consistency
4. **API calls:** Use the built-in fetch utilities in `src/lib/apis/`
5. **State management:** Use Svelte stores for global state

## Troubleshooting

**Build fails:**
- Check Node.js version (requires 18+)
- Clear cache: `rm -rf node_modules .svelte-kit && npm install`

**Docker build slow:**
- Use `.dockerignore` to exclude `node_modules`
- Build incrementally during development

**Changes not reflecting:**
- Clear browser cache
- Rebuild Docker image: `docker-compose build --no-cache open-webui`
