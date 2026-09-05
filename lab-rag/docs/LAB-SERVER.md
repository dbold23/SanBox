# Running LabRAG for the whole lab

One machine runs two things: `labrag serve` (the web page) and `labrag index --every 30`
(keeps the index fresh). Everyone else opens the web page. This document is the copy-paste
setup for keeping both running after a reboot.

## Which machine

- It must stay on and be reachable on the lab network.
- It must see the papers: the NAS share mounted, or Drive for Desktop installed, or a
  service account for Drive (see [GOOGLE-DRIVE.md](GOOGLE-DRIVE.md)).
- A Mac mini or any desktop is plenty. Indexing a thousand PDFs takes tens of minutes the
  first time and seconds afterwards. If Ollama is going to write the answers, 16 GB RAM.
- A NAS that runs Docker or Python works too, as long as it can reach the model API or an
  Ollama server.

Find the machine's address with `hostname` (macOS/Linux) or `hostname` in PowerShell. Lab
members use `http://<that name>.local:8008` or `http://<its IP>:8008`. Consider giving it a
fixed IP in the router.

## macOS (launchd)

Two files in `~/Library/LaunchAgents/`. Replace `/Users/lab` with the real home folder and
check `which labrag` for the real path.

`~/Library/LaunchAgents/edu.csumb.labrag.serve.plist`
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>edu.csumb.labrag.serve</string>
  <key>ProgramArguments</key><array><string>/Users/lab/.local/bin/labrag</string><string>serve</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/lab/.labrag/serve.log</string>
  <key>StandardErrorPath</key><string>/Users/lab/.labrag/serve.log</string>
</dict></plist>
```

`~/Library/LaunchAgents/edu.csumb.labrag.index.plist`
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>edu.csumb.labrag.index</string>
  <key>ProgramArguments</key><array><string>/Users/lab/.local/bin/labrag</string><string>index</string><string>--every</string><string>30</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/lab/.labrag/index.log</string>
  <key>StandardErrorPath</key><string>/Users/lab/.labrag/index.log</string>
</dict></plist>
```

```bash
launchctl load ~/Library/LaunchAgents/edu.csumb.labrag.serve.plist
launchctl load ~/Library/LaunchAgents/edu.csumb.labrag.index.plist
```

Set the Mac to not sleep (System Settings → Energy) and to log in automatically, so the
agents start after a power cut.

## Linux (systemd)

`/etc/systemd/system/labrag-serve.service`
```ini
[Unit]
Description=LabRAG web page
After=network-online.target remote-fs.target

[Service]
User=lab
ExecStart=/home/lab/.local/bin/labrag serve
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/labrag-index.service`
```ini
[Unit]
Description=LabRAG index updater
After=network-online.target remote-fs.target

[Service]
User=lab
ExecStart=/home/lab/.local/bin/labrag index --every 30
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now labrag-serve labrag-index
journalctl -u labrag-index -f        # watch it work
```

`remote-fs.target` makes sure the NAS mount is up first. If the share is not mounted at
index time LabRAG sees an empty folder and refuses to remove anything, so a late mount is
safe; it simply indexes on the next pass.

## Windows

Open **Task Scheduler → Create Task**. Trigger: *At startup*. Action: *Start a program*,
program `labrag`, arguments `serve`. Tick *Run whether user is logged on or not*. Make a
second task with arguments `index --every 30`. Or, simpler, put two shortcuts in the
Startup folder (`shell:startup`) that run `labrag serve` and `labrag index --every 30`.

## Docker

A `Dockerfile` ships with the project (it runs the index updater and the web page in one
container):

```bash
docker build -t labrag ./lab-rag
docker run -d --name labrag -p 8008:8008 \
  -v /mnt/nas/Papers:/papers:ro \
  -v labrag-data:/data \
  -v labrag-models:/root/.labrag/models \
  -e ANTHROPIC_API_KEY=... labrag
```

`/papers` is the (read-only) papers folder, `/data` holds the index, and the models volume
keeps the downloaded embedding model across rebuilds. Add `-e LABRAG_PASSWORD=...` to
protect the page. On a Synology or QNAP NAS, Container Manager / Container Station can run
the same image with the shared folder mounted at `/papers`.

## Passwords and networks

The page has no accounts. On a lab network behind the campus firewall that is usually fine.
If the machine is reachable from outside, set `LABRAG_PASSWORD` (any user name, that
password), or put it behind Tailscale or the campus VPN. Do not port-forward it to the
internet without a password.

## Backups

The index is one file, `labrag.db`, in `LABRAG_DATA`, plus the Drive mirror in
`LABRAG_DATA/drive/`. It can always be rebuilt from the papers with `labrag index --rebuild`,
so it needs no backup of its own. The papers do; that is what the NAS is for.

## Upgrading

```bash
pipx upgrade lab-rag        # or: pip install --upgrade ./lab-rag
labrag doctor
```

If a release changes the embedding model, `labrag index` will say the index was built with a
different embedder; run `labrag index --rebuild` once.
