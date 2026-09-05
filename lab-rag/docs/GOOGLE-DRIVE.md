# Google Drive

There are two ways to get papers from the lab's Google Drive into LabRAG. Pick one.

| | Drive for Desktop (a folder on disk) | Direct sync (LabRAG talks to Drive) |
|---|---|---|
| Works on | macOS, Windows | Anything that runs Python, including a Linux server or NAS |
| Setup | Install the Drive app, tick "Available offline" | 10 minutes in the Google Cloud console, once |
| Configure | `LABRAG_FOLDERS` | `LABRAG_DRIVE_FOLDER` + one credentials file |
| Google Docs | Not indexed (they are links, not files) | Exported as text and indexed |

## Option 1: Google Drive for Desktop

1. Install [Google Drive for Desktop](https://www.google.com/drive/download/) on the LabRAG
   machine and sign in with the account that can see the lab folder.
2. Right-click the lab's papers folder in Drive and choose **Available offline** (otherwise
   files are placeholders and every index run downloads them again).
3. Find the folder on disk:
   - macOS: `~/Library/CloudStorage/GoogleDrive-<you>@csumb.edu/Shared drives/<Drive name>/Papers`
     or `.../My Drive/Papers`. Folders shared *with* you are under **Shared with me**; add a
     shortcut to them in My Drive so they appear on disk.
   - Windows: `G:\Shared drives\<Drive name>\Papers` or `G:\My Drive\Papers`.
4. Put that path in `LABRAG_FOLDERS` (run `labrag init` or edit `~/.labrag/labrag.env`).

That is all. Drive is now just another folder.

## Option 2: Direct sync with a service account

Best for a Linux box or the NAS. LabRAG downloads new and changed files into
`$LABRAG_DATA/drive/`, mirroring the folder structure, and indexes that mirror. Nothing is
ever written to Drive (the credentials are read-only).

1. Go to <https://console.cloud.google.com/>, create a project (any name, e.g. `labrag`).
2. **APIs & Services → Library**: enable **Google Drive API**.
3. **IAM & Admin → Service Accounts → Create service account**. Name it `labrag`. No roles
   are needed. Open it, **Keys → Add key → Create new key → JSON**. A file downloads.
4. Move that file somewhere safe on the LabRAG machine, e.g. `~/.labrag/google-service-account.json`.
   It is a secret: do not commit it, do not put it on the NAS.
5. In Google Drive, open the lab's papers folder, click **Share**, and add the service account's
   e-mail address (it looks like `labrag@<project>.iam.gserviceaccount.com`) as **Viewer**.
   For a Shared drive, add it as a member of the Shared drive instead.
6. Configure LabRAG:
   ```
   LABRAG_DRIVE_FOLDER=https://drive.google.com/drive/folders/1AbC...   # the folder link
   LABRAG_GOOGLE_SERVICE_ACCOUNT=/home/lab/.labrag/google-service-account.json
   ```
7. `labrag doctor` should now report the number of indexable files in the folder.
   `labrag index` syncs the folder first, then indexes.

What gets synced: PDF, DOCX, TXT, Markdown and HTML files, plus Google Docs and Slides
(exported as plain text). Images, videos, spreadsheets and everything else are skipped.
Shortcuts to files and folders are followed. Deleting a file in Drive removes it from the
index at the next run.

## Option 3: Direct sync with your own Google account (OAuth)

Same result as option 2, but the sync runs as you instead of as a service account. Use this
on a personal laptop; it needs a browser on the machine the first time.

1. In the Cloud console project, **APIs & Services → Credentials → Create credentials →
   OAuth client ID → Desktop app**. If asked, configure the consent screen first (internal,
   any name). Download the JSON.
2. ```
   LABRAG_DRIVE_FOLDER=<folder link>
   LABRAG_GOOGLE_CLIENT_SECRET=/path/to/client_secret.json
   ```
3. The first `labrag index` opens a browser for sign-in. The token is saved to
   `~/.labrag/google_token.json` and refreshed automatically afterwards.

## Option 4: rclone

If the lab already uses [rclone](https://rclone.org/drive/), keep using it:

```
rclone sync gdrive:Papers /Volumes/LabNAS/Papers/from-drive
```

on a schedule, and put `/Volumes/LabNAS/Papers` in `LABRAG_FOLDERS`.
