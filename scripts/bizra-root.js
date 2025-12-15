#!/usr/bin/env node
/**
 * BIZRA Root Resolver (Node.js)
 * Returns canonical repo root (works from any subfolder)
 * Usage: const root = execSync('node scripts/bizra-root.js').toString().trim();
 */

const { execSync } = require("node:child_process");
const { existsSync } = require("node:fs");
const path = require("node:path");

function gitRoot() {
    try {
        return execSync("git rev-parse --show-toplevel", {
            stdio: ["ignore", "pipe", "ignore"],
            encoding: "utf8"
        }).trim();
    } catch {
        return null;
    }
}

function walkRoot() {
    let dir = process.cwd();
    while (true) {
        if (existsSync(path.join(dir, ".git"))) {
            return dir;
        }
        const parent = path.dirname(dir);
        if (parent === dir) {
            throw new Error("BIZRA root not found (no .git directory in hierarchy)");
        }
        dir = parent;
    }
}

const root = gitRoot() ?? walkRoot();
console.log(root);
