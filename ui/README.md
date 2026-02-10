# Custom Open WebUI

This directory contains the customized Open WebUI source code.

## Setup

1. Clone Open WebUI source:
```bash
git clone https://github.com/open-webui/open-webui.git ui-custom/open-webui
cd ui-custom/open-webui
```

2. Customize the code:
- Frontend: Modify files in `src/`
- Backend: Modify files in `backend/`
- Styling: Edit `src/lib/styles/` or tailwind config

## Common Customizations

### Change App Name and Branding
- Edit `src/lib/constants.ts` - change app name
- Edit `src/lib/assets/` - replace logo/favicon
- Edit `src/routes/+layout.svelte` - modify layout

### Modify Chat Interface
- Edit `src/lib/components/chat/` - chat components
- Edit `src/lib/components/chat/Messages.svelte` - message display
- Edit `src/lib/components/chat/MessageInput.svelte` - input box

### Add Custom Features
- Add new routes in `src/routes/`
- Add new components in `src/lib/components/`
- Modify backend API in `backend/open_webui/`

### Customize Theme/Colors
- Edit `tailwind.config.js` - modify color scheme
- Edit `src/app.css` - global styles

## Build

After customizing, build with Docker:
```bash
docker-compose build open-webui
docker-compose up open-webui -d
```

## Development Mode

For faster iteration:
```bash
cd ui-custom/open-webui
npm install
npm run dev
```

Frontend will run on http://localhost:5173
Backend runs separately (see backend README)
