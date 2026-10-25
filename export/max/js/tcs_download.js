const path = require('path');
const fs = require('fs');
const { once } = require('events');
const Max = require('max-api');

// ----------------------------------------------------------------------------
// Download a model from the HuggingFace Hub.
//
// Messages IN:
//   download <repo>                  download every file in the repo
//   download <repo> <file> [<file>…] download only the named file(s)
//   destination <dir>                where to save (default: this script's folder)
//   token <hf_token>                 access token for gated/private repos
//   revision <branch-or-commit>      defaults to "main"
//
// Lists OUT (first atom is the selector — route with:
//   route progress status error ):
//   progress <doneBytes> <total>    
//   status <text>
//   error <text>
// ----------------------------------------------------------------------------

// Save next to this script (the help/ folder, alongside the .maxhelp + .pte files).
let destDir = __dirname;
let accessToken = undefined;
let revision = 'main';
let busy = false; // true while a download is running, to block concurrent ones

Max.post(`Loaded ${path.basename(__filename)}. Send: download <repo> [file ...]. Destination: ${destDir}.`);

// @huggingface/hub is installed at runtime by the patch's
// "script npm install @huggingface/hub" message, so it may be missing on first
// run — load it lazily and fail gracefully rather than crashing the script.
let hub = null;
function getHub() {
	if (hub) return hub;
	try {
		hub = require('@huggingface/hub');
	} catch (err) {
		Max.error('@huggingface/hub is not installed — run "npm install @huggingface/hub" first.');
		throw err;
	}
	return hub;
}

Max.addHandler('destination', (dir) => {
	destDir = String(dir);
	Max.post(`Destination: ${destDir}`);
});

Max.addHandler('token', (tok) => {
	accessToken = String(tok);
	Max.post('Access token set');
});

Max.addHandler('revision', (rev) => {
	revision = String(rev);
	Max.post(`Revision: ${revision}`);
});

Max.addHandler('download', async (repo, ...files) => {
	if (repo === undefined) {
		Max.error('Download: missing repo. Usage: download <repo> [file ...]');
		return;
	}
	if (busy) {
		Max.outlet('status', 'A download is already in progress, wait for it to finish to start a new one.');
		Max.post('A download is already in progress, wait for it to finish to start a new one.');
		return;
	}
	busy = true;
	try {
		await runDownload(String(repo), files.map(String));
	} catch (err) {
		const msg = (err && err.message) ? err.message : String(err);
		Max.outlet('error', msg);
		Max.error(`Download failed: ${(err && err.stack) ? err.stack : msg}`);
	} finally {
		busy = false;
	}
});

async function runDownload(repo, files) {
	const { listFiles, downloadFile } = getHub();

	Max.outlet('status', `Listing ${repo}`);

	// One metadata pass to learn every file path and its size, so the overall
	// total is known up front (lets the patch detect completion via done==total).
	const sizes = new Map();
	for await (const entry of listFiles({ repo, recursive: true, revision, accessToken })) {
		if (entry.type === 'file') sizes.set(entry.path, entry.size || 0);
	}

	let targets;
	if (files.length > 0) {
		targets = files.map((p) => ({ path: p, size: sizes.get(p) || 0 }));
		const missing = files.filter((p) => !sizes.has(p));
		if (missing.length) Max.post(`warning: not found in repo listing: ${missing.join(', ')}`);
	} else {
		targets = [...sizes.entries()].map(([p, size]) => ({ path: p, size }));
	}

	if (targets.length === 0) throw new Error('no files to download');

	const totalBytes = targets.reduce((sum, t) => sum + t.size, 0);
	Max.post(`Downloading ${targets.length} file(s) from ${repo}, ~${mb(totalBytes)} MB`);

	let doneBytes = 0;
	for (const target of targets) {
		Max.outlet('status', `Downloading ${target.path}`);
		doneBytes += await downloadOne(downloadFile, repo, target, doneBytes, totalBytes);
	}

	Max.outlet('progress', totalBytes, totalBytes);
	Max.outlet('status', 'Download completed: ' + targets.map((t) => t.path).join(', '));
	Max.post('Download complete.');
}

async function downloadOne(downloadFile, repo, target, baseBytes, totalBytes) {
	const blob = await downloadFile({ repo, path: target.path, revision, accessToken });
	if (!blob) throw new Error(`file not found: ${target.path}`);

	// Fall back to the blob's own size if the listing didn't report one, so the
	// reported total stays consistent for single-file downloads.
	if (!target.size && blob.size) {
		totalBytes += blob.size;
	}

	const outPath = path.join(destDir, target.path);
	fs.mkdirSync(path.dirname(outPath), { recursive: true });

	const ws = fs.createWriteStream(outPath);
	const reader = blob.stream().getReader();
	let fileBytes = 0;
	let lastEmit = 0;

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			const chunk = Buffer.from(value);
			// Respect backpressure so multi-GB files don't pile up in memory.
			if (!ws.write(chunk)) await once(ws, 'drain');
			fileBytes += chunk.length;

			const now = Date.now();
			if (now - lastEmit > 200) {
				Max.outlet('progress', baseBytes + fileBytes, totalBytes);
				lastEmit = now;
			}
		}
	} finally {
		reader.releaseLock();
	}

	await new Promise((resolve, reject) => ws.end((err) => (err ? reject(err) : resolve())));
	return fileBytes;
}

function mb(bytes) {
	return (bytes / 1e6).toFixed(1);
}
