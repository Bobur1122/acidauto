# AcidAuto Render

The game logic and the protected browser control panel live in `app.py`. The small Render manifest and dependency list are needed to deploy it; Telegram login data and calibration are never committed.

## Telegram login

1. Create the Render Web Service from this repository and set a strong `ADMIN_TOKEN` environment variable in Render.
2. Open the service URL on a computer and enter that token.
3. The control page shows the Telegram Web login screen. Scan its QR code with the Telegram app on your phone.
4. Use the image on the control page to open `@acidfirm_bot` and launch the game. Press **Start bot** once the game is visible.
5. Keep the browser profile on persistent storage. The profile is an active Telegram account session; never upload it to GitHub or share it.

The service uses the profile directory under `DATA_DIR/telegram-profile`. The login does not require a password in the source code. The QR scan is a one-time user action, and Telegram can require a fresh login after revoking or expiring a session.

## UptimeRobot and storage

Set an HTTP monitor to `https://YOUR-SERVICE.onrender.com/health` (every 5 minutes is suitable for an idle monitor). Render's Free Web Service can spin down after 15 minutes without inbound traffic and can restart at any time. It has no persistent disk, so UptimeRobot may keep it awake but cannot preserve the Telegram session or lead calibration after a restart or deploy.

For reliable 24/7 operation, use a paid Render Web Service with a persistent disk mounted at `/var/data`, then set `DATA_DIR=/var/data`. This is required to preserve the Telegram login between restarts. A paid persistent disk is not included in the free service.

## Deploy

Render reads `render.yaml`. For a free trial, use the included free plan and accept that session state can be lost. For persistent 24/7 use, change the service to a paid plan and attach the disk described above in the Render Dashboard. Set `ADMIN_TOKEN` as a Render environment variable; do not put it in GitHub.

## Limits

This is a server/browser port of the existing screen-based algorithm. Playwright screenshots and clicks have different timing from Windows `mss`/Win32 input, so identical scores cannot be guaranteed until it is tested against the live game. The `IDEAL` and score fields are visual estimates. Start with a monitored test run before leaving it unattended.
